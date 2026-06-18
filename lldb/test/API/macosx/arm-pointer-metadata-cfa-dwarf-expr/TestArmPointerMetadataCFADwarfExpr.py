import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
@skipIf(archs=no_match(["arm64"]))
class TestArmPointerMetadataStripping(TestBase):
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(self, "foo")

        # Step over the first two instructions of foo in order to
        # toggle the bit of fp and save it on the stack:
        # orr   x29, x29, #0x1000000000000000
        # stp	x29, x30, [sp, #-16]!
        # This is effectively adding metadata to the CFA of the caller frame (main).
        thread.StepInstruction(False)
        thread.StepInstruction(False)

        # The location of `argv` has been artificially made equal to the CFA of the frame.
        # As such, it should have the metadata artificially set previously.
        argv_addr = thread.frames[1].GetValueForVariablePath("&argv")
        self.assertTrue(argv_addr.IsValid())
        argv_addr_uint = argv_addr.GetValueAsUnsigned()
        self.assertNotEqual((argv_addr_uint & (1 << 60)), 0)

        # GetCFA strips metadata.
        cfa = thread.frames[1].GetCFA()
        self.assertEqual((cfa & (1 << 60)), 0)

        # If the test worked correctly, the cfa and the location should be identical,
        # modulo the metadata.
        self.assertEqual(cfa | (1 << 60), argv_addr_uint)
