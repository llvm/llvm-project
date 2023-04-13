""" Check that register fields found in target XML are properly processed.

These tests make XML out of string substitution. This can lead to some strange
failures. Check that the final XML is valid and each child is indented more than
the parent tag.
"""

from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class MultiDocResponder(MockGDBServerResponder):
    # docs is a dictionary of filename -> file content.
    def __init__(self, docs):
        super().__init__()
        self.docs = docs

    def qXferRead(self, obj, annex, offset, length):
        try:
            return self.docs[annex], False
        except KeyError:
            return None,

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return ''.join([
          # Data for all registers requested by the tests below.
          # 0x7 and 0xE are used because their lsb and msb are opposites, which
          # is needed for a byte order test.
          '77777777EEEEEEEE', # 64 bit x0/r0
          '7777EEEE', # 32 bit cpsr/fpc
          '0000000000000000', # 64 bit pc/pswa
        ])

class TestXMLRegisterFlags(GDBRemoteTestBase):
    def setup_multidoc_test(self, docs):
        self.server.responder = MultiDocResponder(docs)
        target = self.dbg.CreateTarget('')

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets process")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets process"))

        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateStopped])

    def setup_register_test(self, registers):
        self.setup_multidoc_test(
          # This *must* begin with the opening tag, leading whitespace is not allowed.
          {'target.xml' : dedent("""\
            <?xml version="1.0"?>
              <target version="1.0">
                <architecture>aarch64</architecture>
                <feature name="org.gnu.gdb.aarch64.core">
                  {}
                </feature>
            </target>""").format(registers)})

    def setup_flags_test(self, flags):
        # pc is required here though we don't look at it in the tests.
        # x0 is only used by some tests but always including it keeps the data ordering
        # the same throughout.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            {}
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""".format(
            flags))

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_no_flags(self):
        self.setup_flags_test("")
        self.expect("register read cpsr", substrs=["= 0xeeee7777"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_single_field_pad_msb(self):
        self.setup_flags_test("""<field name="SP" start="0" end="0"/>""")
        # Pads from 31 to 1.
        self.expect("register read cpsr", substrs=["(SP = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_single_field_pad_lsb(self):
        self.setup_flags_test("""<field name="SP" start="31" end="31"/>""")
        self.expect("register read cpsr", substrs=["(SP = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_multiple_fields_sorted(self):
        self.setup_flags_test("""<field name="SP" start="0" end="0"/>
                                 <field name="EL" start="1" end="2"/>""")

        # Fields should be sorted with MSB on the left.
        self.expect("register read cpsr", substrs=["(EL = 3, SP = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_ignore_invalid_start_end(self):
        self.setup_flags_test(
          # Is valid so is used.
          '<field name="EL" start="2" end="3"/>'
          # Start/end cannot be negative, ignored.
          '<field name="SP" start="-1" end="2"/>'
          '<field name="SP2" start="1" end="-5"/>'
          # Start is not <= end, ignored.
          '<field name="ABC" start="12" end="10"/>'
          # Start cannot be >= (size of register in bits)
          '<field name="?" start="32" end="29"/>'
          # End cannot be >= (size of register in bits)
          '<field name="DEF" start="30" end="35"/>')

        self.expect("register read cpsr", substrs=["(EL = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_field_overlap(self):
        self.setup_flags_test(
          '<field name="?" start="10" end="12"/>'
          # A overlaps B
          '<field name="A" start="0" end="3"/>'
          '<field name="B" start="0" end="0"/>')

        # Ignore the whole flags set, it is unlikely to be valid.
        self.expect("register read cpsr", substrs=["("], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_field_required_attributes(self):
        # Fields must have a name, start and end. Any without are ignored.
        self.setup_flags_test(
          # Missing name
          '<field start="0" end="0"/>'
          # Missing start
          '<field name="A" end="0"/>'
          # Missing end
          '<field name="B" start="0"/>'
          # Valid
          '<field name="C" start="0" end="0"/>')

        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_little_endian_target_order(self):
        # We are using little endian AArch64 here.
        self.setup_register_test("""\
           <flags id="cpsr_flags" size="4">
             <field name="lsb" start="0" end="0"/>
             <field name="msb" start="31" end="31"/>
           </flags>
           <flags id="x0_flags" size="8">
             <field name="lsb" start="0" end="0"/>
             <field name="msb" start="63" end="63"/>
           </flags>
           <reg name="pc" bitsize="64"/>
           <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
           <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        # If lldb used the wrong byte ordering for the value for printing fields,
        # these field values would flip. Since the top and bottom bits of 0x7 and 0xE
        # are different.
        self.expect("register read cpsr x0", substrs=[
          "    cpsr = 0xeeee7777\n"
          "         = (msb = 1, lsb = 1)\n"
          "      x0 = 0xeeeeeeee77777777\n"
          "         = (msb = 1, lsb = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    # Unlike AArch64, we do need the backend present for this test to work.
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_target_order(self):
        # s390x/SystemZ is big endian.
        self.setup_multidoc_test({
            'target.xml' : dedent("""\
              <?xml version="1.0"?>
              <target version="1.0">
                <architecture>s390x</architecture>
                <feature name="org.gnu.gdb.s390x.core">
                  <flags id="r0_flags" size="8">
                    <field name="lsb" start="0" end="0"/>
                    <field name="msb" start="63" end="63"/>
                  </flags>
                  <flags id="fpc_flags" size="4">
                    <field name="lsb" start="0" end="0"/>
                    <field name="msb" start="31" end="31"/>
                  </flags>
                  <reg name="r0" bitsize="64" type="r0_flags"/>
                  <reg name="fpc" bitsize="32" type="fpc_flags"/>
                  <reg name="pswa" bitsize="64"/>
                </feature>
              </target>""")})

        # If we did not swap correctly, these fields would show as 1s when run on
        # a little endian host.
        self.expect("register read r0 fpc", substrs=[
          "      r0 = 0x77777777eeeeeeee\n"
          "         = (msb = 0, lsb = 0)\n"
          "     fpc = 0x7777eeee\n"
          "         = (msb = 0, lsb = 0)\n"
        ])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_many_flag_sets(self):
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            <field name="correct" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags_alt" size="4">
            <field name="incorrect" start="0" end="0"/>
          </flags>
          <flags id="x0_flags" size="8">
            <field name="foo" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        self.expect("register read cpsr x0", substrs=[
          "    cpsr = 0xeeee7777\n"
          "         = (correct = 1)\n"
          "      x0 = 0xeeeeeeee77777777\n"
          "         = (foo = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_repeated_flag_set(self):
        # The second definition of "cpsr_flags" should be ignored.
        # This is because we assign the types to registers as we go. If we allowed
        # the later flag set, it would destroy the first definition, making the
        # pointer to the flags invalid.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            <field name="correct" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags" size="4">
            <field name="incorrect" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        self.expect("register read cpsr", substrs=["(correct = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_missing_flags(self):
        self.setup_register_test("""\
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        # Register prints with default formatting only if we can't find the
        # flags type.
        self.expect("register read cpsr", substrs=["cpsr = 0xeeee7777"])
        self.expect("register read cpsr", substrs=["("], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_invalid_size(self):
        # We're not using the size for anything yet so just check that we handle
        # it not being a positive integer.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="???">
            <field name="A" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags" size="-1">
            <field name="B" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags" size="4">
            <field name="C" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        # Only the final set has a valid size, use that.
        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_unknown_attribute(self):
        # Unknown attributes on flags or field are ignored.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4" stuff="abcd">
            <field name="A" start="0" abcd="???" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        self.expect("register read cpsr", substrs=["(A = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_requried_attributes(self):
        # flags must have an id and size so the flags with "C" is the only valid one
        # here.
        self.setup_register_test("""\
          <flags size="4">
            <field name="A" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags">
            <field name="B" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags" size="4">
            <field name="C" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""")

        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_format_set(self):
        # lldb also sends "format". If that is set, we should still read the
        # flags type.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            format="example"/>""")

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_encoding_set(self):
        # lldb also sends "encoding". If that is set, we should still read the
        # flags type.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            encoding="example"/>""")

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_encoding_and_format_set(self):
        # As above but both encoding and format are set.
        self.setup_register_test("""\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            encoding="example" format="example"/>""")

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_multiple_lines(self):
        # Since we use C types they follow lldb's usual decisions as to whether
        # to print them on one line or many. Long field names will usually mean
        # many lines.
        self.setup_flags_test(
          '<field name="this_is_a_long_field_0" start="0" end="0"/>'
          '<field name="this_is_a_long_field_1" start="1" end="1"/>'
          '<field name="this_is_a_long_field_2" start="2" end="2"/>'
          '<field name="this_is_a_long_field_3" start="3" end="3"/>')

        self.expect("register read cpsr", substrs=[
          "    cpsr = 0xeeee7777\n"
          "         = {\n"
          "             this_is_a_long_field_3 = 0\n"
          "             this_is_a_long_field_2 = 1\n"
          "             this_is_a_long_field_1 = 1\n"
          "             this_is_a_long_field_0 = 1\n"
          "           }"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_child_limit(self):
        # Flags print like C types so they should follow the child limit setting.
        self.runCmd("settings set target.max-children-count 3")
        self.setup_flags_test(
          '<field name="field_0" start="0" end="0"/>'
          '<field name="field_1" start="1" end="1"/>'
          '<field name="field_2" start="2" end="2"/>')

        self.expect("register read cpsr", substrs=["= (field_2 = 1, field_1 = 1, ...)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes(self):
        # Certain targets e.g. s390x QEMU split their defintions over multiple
        # files that are included into target.xml.
        self.setup_multidoc_test({
            # The formatting is very specific here. lldb doesn't like leading
            # spaces, and nested tags must be indented more than their parent.
            'target.xml' : dedent("""\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
               </target>"""),
            'core.xml' : dedent("""\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="cpsr_flags" size="4">
                    <field name="B" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
                  <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>
                </feature>
            """),
        })

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes_multiple(self):
        self.setup_multidoc_test({
            'target.xml' : dedent("""\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
                 <xi:include href="core-2.xml"/>
               </target>"""),
            'core.xml' : dedent("""\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="x0_flags" size="4">
                    <field name="B" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
                </feature>"""),
            'core-2.xml' : dedent("""\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="cpsr_flags" size="4">
                    <field name="C" start="0" end="0"/>
                  </flags>
                  <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>
                </feature>
            """),
        })

        self.expect("register read x0 cpsr", substrs=["(B = 1)", "(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes_flags_redefined(self):
        self.setup_multidoc_test({
            'target.xml' : dedent("""\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
                 <xi:include href="core-2.xml"/>
               </target>"""),
            # Treating xi:include as a textual include, my_flags is first defined
            # in core.xml. The second definition in core-2.xml
            # is ignored.
            'core.xml' : dedent("""\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="my_flags" size="4">
                    <field name="correct" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="my_flags"/>
                </feature>"""),
            # The my_flags here is ignored, so cpsr will use the my_flags from above.
            'core-2.xml' : dedent("""\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="my_flags" size="4">
                    <field name="incorrect" start="0" end="0"/>
                  </flags>
                  <reg name="cpsr" regnum="33" bitsize="32" type="my_flags"/>
                </feature>
            """),
        })

        self.expect("register read x0", substrs=["(correct = 1)"])
        self.expect("register read cpsr", substrs=["(correct = 1)"])
