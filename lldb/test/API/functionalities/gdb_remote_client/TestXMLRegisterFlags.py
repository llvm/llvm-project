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
            return (None,)

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return "".join(
            [
                # Data for all registers requested by the tests below.
                # 0x7 and 0xE are used because their lsb and msb are opposites, which
                # is needed for a byte order test.
                "77777777EEEEEEEE",  # 64 bit x0/r0
                "7777EEEE",  # 32 bit cpsr/fpc
                "0000000000000000",  # 64 bit pc/pswa
            ]
        )


class TestXMLRegisterFlags(GDBRemoteTestBase):
    def setup_multidoc_test(self, docs):
        self.server.responder = MultiDocResponder(docs)
        target = self.dbg.CreateTarget("")

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets process")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets process")
            )

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

    def setup_register_test(self, registers):
        self.setup_multidoc_test(
            # This *must* begin with the opening tag, leading whitespace is not allowed.
            {
                "target.xml": dedent(
                    """\
            <?xml version="1.0"?>
              <target version="1.0">
                <architecture>aarch64</architecture>
                <feature name="org.gnu.gdb.aarch64.core">
                  {}
                </feature>
            </target>"""
                ).format(registers)
            }
        )

    def setup_flags_test(self, flags):
        # pc is required here though we don't look at it in the tests.
        # x0 is only used by some tests but always including it keeps the data ordering
        # the same throughout.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            {}
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>""".format(
                flags
            )
        )

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
        self.setup_flags_test(
            """<field name="SP" start="0" end="0"/>
                                 <field name="EL" start="1" end="2"/>"""
        )

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
            '<field name="DEF" start="30" end="35"/>'
        )

        self.expect("register read cpsr", substrs=["(EL = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_field_overlap(self):
        self.setup_flags_test(
            '<field name="?" start="10" end="12"/>'
            # A overlaps B
            '<field name="A" start="0" end="3"/>'
            '<field name="B" start="0" end="0"/>'
        )

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
            '<field name="C" start="0" end="0"/>'
        )

        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_little_endian_target_order(self):
        # We are using little endian AArch64 here.
        self.setup_register_test(
            """\
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
           <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        # If lldb used the wrong byte ordering for the value for printing fields,
        # these field values would flip. Since the top and bottom bits of 0x7 and 0xE
        # are different.
        self.expect(
            "register read cpsr x0",
            substrs=[
                "    cpsr = 0xeeee7777\n"
                "         = (msb = 1, lsb = 1)\n"
                "      x0 = 0xeeeeeeee77777777\n"
                "         = (msb = 1, lsb = 1)"
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    # Unlike AArch64, we do need the backend present for this test to work.
    @skipIfLLVMTargetMissing("SystemZ")
    def test_big_endian_target_order(self):
        # s390x/SystemZ is big endian.
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
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
              </target>"""
                )
            }
        )

        # If we did not swap correctly, these fields would show as 1s when run on
        # a little endian host.
        self.expect(
            "register read r0 fpc",
            substrs=[
                "      r0 = 0x77777777eeeeeeee\n"
                "         = (msb = 0, lsb = 0)\n"
                "     fpc = 0x7777eeee\n"
                "         = (msb = 0, lsb = 0)\n"
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_many_flag_sets(self):
        self.setup_register_test(
            """\
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
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect(
            "register read cpsr x0",
            substrs=[
                "    cpsr = 0xeeee7777\n"
                "         = (correct = 1)\n"
                "      x0 = 0xeeeeeeee77777777\n"
                "         = (foo = 1)"
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_repeated_flag_set(self):
        # The second definition of "cpsr_flags" should be ignored.
        # This is because we assign the types to registers as we go. If we allowed
        # the later flag set, it would destroy the first definition, making the
        # pointer to the flags invalid.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            <field name="correct" start="0" end="0"/>
          </flags>
          <flags id="cpsr_flags" size="4">
            <field name="incorrect" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register read cpsr", substrs=["(correct = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_missing_flags(self):
        self.setup_register_test(
            """\
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        # Register prints with default formatting only if we can't find the
        # flags type.
        self.expect("register read cpsr", substrs=["cpsr = 0xeeee7777"])
        self.expect("register read cpsr", substrs=["("], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_invalid_size(self):
        # We're not using the size for anything yet so just check that we handle
        # it not being a positive integer.
        self.setup_register_test(
            """\
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
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        # Only the final set has a valid size, use that.
        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_unknown_attribute(self):
        # Unknown attributes on flags or field are ignored.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4" stuff="abcd">
            <field name="A" start="0" abcd="???" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register read cpsr", substrs=["(A = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_required_attributes(self):
        # flags must have an id and size so the flags with "C" is the only valid one
        # here.
        self.setup_register_test(
            """\
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
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register read cpsr", substrs=["(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_register_size_mismatch(self):
        # If the size of the flag set found does not match the size of the
        # register, we discard the flags.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="8">
            <field name="C" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register read cpsr", substrs=["(C = 1)"], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_format_set(self):
        # lldb also sends "format". If that is set, we should still read the
        # flags type.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            format="example"/>"""
        )

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_encoding_set(self):
        # lldb also sends "encoding". If that is set, we should still read the
        # flags type.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            encoding="example"/>"""
        )

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_set_even_if_encoding_and_format_set(self):
        # As above but both encoding and format are set.
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            <field name="B" start="0" end="0"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"
            encoding="example" format="example"/>"""
        )

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
            '<field name="this_is_a_long_field_3" start="3" end="3"/>'
        )

        self.expect(
            "register read cpsr",
            substrs=[
                "    cpsr = 0xeeee7777\n"
                "         = {\n"
                "             this_is_a_long_field_3 = 0\n"
                "             this_is_a_long_field_2 = 1\n"
                "             this_is_a_long_field_1 = 1\n"
                "             this_is_a_long_field_0 = 1\n"
                "           }"
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_child_limit(self):
        # Flags print like C types so they should follow the child limit setting.
        self.runCmd("settings set target.max-children-count 3")
        self.setup_flags_test(
            '<field name="field_0" start="0" end="0"/>'
            '<field name="field_1" start="1" end="1"/>'
            '<field name="field_2" start="2" end="2"/>'
        )

        self.expect("register read cpsr", substrs=["= (field_2 = 1, field_1 = 1, ...)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_format_disables_flags(self):
        # If asked for a specific format, don't print flags after it.
        self.setup_flags_test('<field name="field_0" start="0" end="0"/>')

        self.expect("register read cpsr --format X", substrs=["cpsr = 0xEEEE7777"])
        self.expect(
            "register read cpsr --format X", substrs=["field_0"], matching=False
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes(self):
        # Certain targets e.g. s390x QEMU split their defintions over multiple
        # files that are included into target.xml.
        self.setup_multidoc_test(
            {
                # The formatting is very specific here. lldb doesn't like leading
                # spaces, and nested tags must be indented more than their parent.
                "target.xml": dedent(
                    """\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
               </target>"""
                ),
                "core.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="cpsr_flags" size="4">
                    <field name="B" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
                  <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>
                </feature>
            """
                ),
            }
        )

        self.expect("register read cpsr", substrs=["(B = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes_multiple(self):
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
                 <xi:include href="core-2.xml"/>
               </target>"""
                ),
                "core.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="x0_flags" size="8">
                    <field name="B" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
                </feature>"""
                ),
                "core-2.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="cpsr_flags" size="4">
                    <field name="C" start="0" end="0"/>
                  </flags>
                  <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>
                </feature>
            """
                ),
            }
        )

        self.expect("register read x0 cpsr", substrs=["(B = 1)", "(C = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_xml_includes_flags_redefined(self):
        self.setup_multidoc_test(
            {
                "target.xml": dedent(
                    """\
               <?xml version="1.0"?>
               <target version="1.0">
                 <architecture>aarch64</architecture>
                 <xi:include href="core.xml"/>
                 <xi:include href="core-2.xml"/>
               </target>"""
                ),
                # Treating xi:include as a textual include, my_flags is first defined
                # in core.xml. The second definition in core-2.xml
                # is ignored.
                "core.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="my_flags" size="8">
                    <field name="correct" start="0" end="0"/>
                  </flags>
                  <reg name="pc" bitsize="64"/>
                  <reg name="x0" regnum="0" bitsize="64" type="my_flags"/>
                </feature>"""
                ),
                # The my_flags here is ignored, so x1 will use the my_flags from above.
                "core-2.xml": dedent(
                    """\
                <?xml version="1.0"?>
                <feature name="org.gnu.gdb.aarch64.core">
                  <flags id="my_flags" size="8">
                    <field name="incorrect" start="0" end="0"/>
                  </flags>
                  <reg name="x1" regnum="33" bitsize="64" type="my_flags"/>
                </feature>
            """
                ),
            }
        )

        self.expect("register read x0", substrs=["(correct = 1)"])
        self.expect("register read x1", substrs=["(correct = 1)"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_in_register_info(self):
        # See RegisterFlags for comprehensive formatting tests.
        self.setup_flags_test(
            '<field name="D" start="0" end="7"/>'
            '<field name="C" start="8" end="15"/>'
            '<field name="B" start="16" end="23"/>'
            '<field name="A" start="24" end="31"/>'
        )

        # The table should split according to terminal width.
        self.runCmd("settings set term-width 17")

        self.expect(
            "register info cpsr",
            substrs=[
                "       Name: cpsr\n"
                "       Size: 4 bytes (32 bits)\n"
                "    In sets: general (index 0)\n"
                "\n"
                "| 31-24 | 23-16 |\n"
                "|-------|-------|\n"
                "|   A   |   B   |\n"
                "\n"
                "| 15-8 | 7-0 |\n"
                "|------|-----|\n"
                "|  C   |  D  |"
            ],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_flags_name_xml_reserved_characters(self):
        """Check that lldb converts reserved character replacements like &amp;
        when found in field names."""
        self.setup_flags_test(
            '<field name="E&amp;" start="0" end="0"/>'
            '<field name="D&quot;" start="1" end="1"/>'
            '<field name="C&apos;" start="2" end="2"/>'
            '<field name="B&gt;" start="3" end="3"/>'
            '<field name="A&lt;" start="4" end="4"/>'
        )

        self.expect(
            "register info cpsr",
            substrs=["| A< | B> | C' | D\" | E& |"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_no_enum(self):
        """Check that lldb does not try to print an enum when there isn't one."""

        self.setup_flags_test('<field name="E" start="0" end="0">' "</field>")

        self.expect("register info cpsr", patterns=["E:.*$"], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_type_not_found(self):
        """Check that lldb uses the default format if we don't find the enum type."""
        self.setup_register_test(
            """\
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register read cpsr", patterns=["\(E = 1\)$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_duplicated_evalue(self):
        """Check that lldb only uses the last instance of a evalue for each
        value."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="abc" value="1"/>
            <evalue name="def" value="1"/>
            <evalue name="geh" value="2"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="1" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 1 = def, 2 = geh$"])
        self.expect("register read cpsr", patterns=["\(E = def \| geh\)$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_duplicated(self):
        """Check that lldb only uses the last instance of enums with the same
        id."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="abc" value="1"/>
          </enum>
          <enum id="some_enum" size="4">
            <evalue name="def" value="1"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 1 = def$"])
        self.expect("register read cpsr", patterns=["\(E = def\)$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_use_first_valid(self):
        """Check that lldb uses the first enum that parses correctly and ignores
        the rest."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4"/>
          <enum size="4">
            <evalue name="invalid" value="1"/>
          </enum>
          <enum id="some_enum" size="4">
            <evalue name="valid" value="1"/>
          </enum>
          <enum id="another_enum" size="4">
            <evalue name="invalid" value="1"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 1 = valid$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_empty_name(self):
        """Check that lldb ignores evalues with an empty name."""

        # The only potential use case for empty names is to shadow an evalue
        # declared later so that it's name is hidden should the debugger only
        # pick one of them. This behaviour would be debugger specific so the protocol
        # would probably not care or leave it up to us, and I think it's not a
        # useful thing to allow.

        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="" value="1"/>
            <evalue name="valid" value="2"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="1" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 2 = valid$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_invalid_value(self):
        """Check that lldb ignores evalues with an invalid value."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="negative_dec" value="-1"/>
            <evalue name="negative_hex" value="-0x1"/>
            <evalue name="negative_bin" value="-0b1"/>
            <evalue name="negative_float" value="-0.5"/>
            <evalue name="nan" value="aardvark"/>
            <evalue name="dec" value="1"/>
            <evalue name="hex" value="0x2"/>
            <evalue name="octal" value="03"/>
            <evalue name="float" value="0.5"/>
            <evalue name="bin" value="0b100"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="2" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect(
            "register info cpsr", patterns=["E: 1 = dec, 2 = hex, 3 = octal, 4 = bin$"]
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_out_of_range(self):
        """Check that lldb will not use an enum type if one of its evalues
        exceeds the size of the field it is applied to."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="A" value="0"/>
            <evalue name="B" value="2"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        # The whole eunm is rejected even if just 1 value is out of range.
        self.expect("register info cpsr", patterns=["E: 0 = "], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_ignore_unknown_attributes(self):
        """Check that lldb ignores unknown attributes on an enum or evalue."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4" foo=\"bar\">
            <evalue name="valid" value="1" colour=\"red"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 1 = valid$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_required_attributes(self):
        """Check that lldb rejects any evalue missing a name and/or value."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="foo"/>
            <evalue value="1"/>
            <evalue />
            <evalue name="valid" value="1"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="0" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect("register info cpsr", patterns=["E: 1 = valid$"])

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_name_xml_reserved_characters(self):
        """Check that lldb converts reserved character replacements like &amp;
        when found in evalue names."""
        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="A&amp;"  value="0"/>
            <evalue name="B&quot;" value="1"/>
            <evalue name="C&apos;" value="2"/>
            <evalue name="D&gt;"   value="3"/>
            <evalue name="E&lt;"   value="4"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="E" start="0" end="2" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        self.expect(
            "register info cpsr",
            patterns=["E: 0 = A&, 1 = B\", 2 = C', 3 = D>, 4 = E<$"],
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_enum_value_range(self):
        """Check that lldb ignores enums whose values would not fit into
        their field."""

        self.setup_register_test(
            """\
          <enum id="some_enum" size="4">
            <evalue name="A" value="0"/>
            <evalue name="B" value="1"/>
            <evalue name="C" value="2"/>
            <evalue name="D" value="3"/>
            <evalue name="E" value="4"/>
          </enum>
          <flags id="cpsr_flags" size="4">
            <field name="foo" start="0" end="1" type="some_enum"/>
            <field name="bar" start="2" end="10" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        # some_enum can apply to foo
        self.expect(
            "register info cpsr", patterns=["bar: 0 = A, 1 = B, 2 = C, 3 = D, 4 = E$"]
        )
        # but not to bar
        self.expect("register info cpsr", patterns=["foo: "], matching=False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_evalue_value_limits(self):
        """Check that lldb can handle an evalue for a field up to 64 bits
        in size and anything greater is ignored."""

        self.setup_register_test(
            """\
          <enum id="some_enum" size="8">
            <evalue name="min" value="0"/>
            <evalue name="max" value="0xffffffffffffffff"/>
            <evalue name="invalid" value="0xfffffffffffffffff"/>
          </enum>
          <flags id="x0_flags" size="8">
            <field name="foo" start="0" end="63" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32"/>"""
        )

        self.expect(
            "register info x0", patterns=["foo: 0 = min, 18446744073709551615 = max$"]
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_field_size_limit(self):
        """Check that lldb ignores any field > 64 bits. We can't handle those
        correctly."""

        self.setup_register_test(
            """\
          <flags id="x0_flags" size="8">
            <field name="invalid" start="0" end="64"/>
            <field name="valid" start="0" end="63"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32"/>"""
        )

        self.expect(
            "register info x0", substrs=["| 63-0  |\n" "|-------|\n" "| valid |"]
        )

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_many_fields_same_enum(self):
        """Check that an enum can be reused by many fields, and fields of many
        registers."""

        self.setup_register_test(
            """\
          <enum id="some_enum" size="8">
            <evalue name="valid" value="1"/>
          </enum>
          <flags id="x0_flags" size="8">
            <field name="f1" start="0" end="0" type="some_enum"/>
            <field name="f2" start="1" end="1" type="some_enum"/>
          </flags>
          <flags id="cpsr_flags" size="4">
            <field name="f1" start="0" end="0" type="some_enum"/>
            <field name="f2" start="1" end="1" type="some_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32" type="cpsr_flags"/>"""
        )

        expected_info = [
            dedent(
                """\
             f2: 1 = valid

             f1: 1 = valid$"""
            )
        ]
        self.expect("register info x0", patterns=expected_info)

        self.expect("register info cpsr", patterns=expected_info)

        expected_read = ["\(f2 = valid, f1 = valid\)$"]
        self.expect("register read x0", patterns=expected_read)
        self.expect("register read cpsr", patterns=expected_read)

    @skipIfXmlSupportMissing
    @skipIfRemote
    def test_fields_same_name_different_enum(self):
        """Check that lldb does something sensible when there are two fields with
        the same name, but their enum types differ."""

        # It's unlikely anyone would do this intentionally but it is allowed by
        # the protocol spec so we have to cope with it.
        self.setup_register_test(
            """\
          <enum id="foo_enum" size="8">
            <evalue name="foo_0" value="1"/>
          </enum>
          <enum id="foo_alt_enum" size="8">
            <evalue name="foo_1" value="1"/>
          </enum>
          <flags id="x0_flags" size="8">
            <field name="foo" start="0" end="0" type="foo_enum"/>
            <field name="foo" start="1" end="1" type="foo_alt_enum"/>
          </flags>
          <reg name="pc" bitsize="64"/>
          <reg name="x0" regnum="0" bitsize="64" type="x0_flags"/>
          <reg name="cpsr" regnum="33" bitsize="32"/>"""
        )

        self.expect(
            "register info x0",
            patterns=[
                dedent(
                    """\
             foo: 1 = foo_1

             foo: 1 = foo_0$"""
                )
            ],
        )

        self.expect("register read x0", patterns=["\(foo = foo_1, foo = foo_0\)$"])
