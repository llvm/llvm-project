"""
Test that memory tagging features work with Linux core files.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class AArch64LinuxMTEMemoryTagCoreFileTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    MTE_BUF_ADDR = hex(0xFFFF82C74000)
    BUF_ADDR = hex(0xFFFF82C73000)

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_tag_core_file_memory_region(self):
        """Test that memory regions are marked as tagged when there is a tag
        segment in the core file."""
        self.runCmd("target create --core core.mte")

        # There should only be one tagged region.
        self.runCmd("memory region --all")
        got = self.res.GetOutput()
        found_tagged_region = False

        for line in got.splitlines():
            if "memory tagging: enabled" in line:
                if found_tagged_region:
                    self.fail("Expected only one tagged region.")
                found_tagged_region = True

        self.assertTrue(found_tagged_region, "Did not find a tagged memory region.")

        # mte_buf is tagged, buf is not.
        tagged = "memory tagging: enabled"
        self.expect("memory region {}".format(self.MTE_BUF_ADDR), patterns=[tagged])
        self.expect(
            "memory region {}".format(self.BUF_ADDR), patterns=[tagged], matching=False
        )

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_tag_core_file_tag_write(self):
        """Test that "memory tag write" does not work with core files
        as they are read only."""
        self.runCmd("target create --core core.mte")

        self.expect(
            "memory tag write {} 1".format(self.MTE_BUF_ADDR),
            error=True,
            patterns=["error: elf-core does not support writing memory tags"],
        )

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_tag_core_file_tag_read(self):
        """Test that "memory tag read" works with core files."""
        self.runCmd("target create --core core.mte")

        # Tags are packed 2 per byte meaning that in addition to granule alignment
        # there is also 2 x granule alignment going on.

        # All input validation should work as normal.
        not_tagged_pattern = (
            "error: Address range 0x[A-Fa-f0-9]+:0x[A-Fa-f0-9]+ "
            "is not in a memory tagged region"
        )
        self.expect(
            "memory tag read {}".format(self.BUF_ADDR),
            error=True,
            patterns=[not_tagged_pattern],
        )
        # The first part of this range is not tagged.
        self.expect(
            "memory tag read {addr}-16 {addr}+16".format(addr=self.MTE_BUF_ADDR),
            error=True,
            patterns=[not_tagged_pattern],
        )
        # The last part of this range is not tagged.
        self.expect(
            "memory tag read {addr}+4096-16 {addr}+4096+16".format(
                addr=self.MTE_BUF_ADDR
            ),
            error=True,
            patterns=[not_tagged_pattern],
        )

        self.expect(
            "memory tag read {addr}+16 {addr}".format(addr=self.MTE_BUF_ADDR),
            error=True,
            patterns=[
                "error: End address \(0x[A-Fa-f0-9]+\) "
                "must be greater than the start address "
                "\(0x[A-Fa-f0-9]+\)"
            ],
        )

        # The simplest scenario. 2 granules means 1 byte of packed tags
        # with no realignment required.
        self.expect(
            "memory tag read {addr} {addr}+32".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)$"
            ],
        )

        # Here we want just one tag so must use half of the first byte.
        # (start is aligned length is not)
        self.expect(
            "memory tag read {addr} {addr}+16".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n" "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0$"
            ],
        )
        # Get the other half of the first byte.
        # (end is aligned start is not)
        self.expect(
            "memory tag read {addr}+16 {addr}+32".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)$"
            ],
        )

        # Same thing but with a starting range > 1 granule.
        self.expect(
            "memory tag read {addr} {addr}+48".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+20, 0x[A-Fa-f0-9]+30\): 0x2 \(mismatch\)$"
            ],
        )
        self.expect(
            "memory tag read {addr}+16 {addr}+64".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+20, 0x[A-Fa-f0-9]+30\): 0x2 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+30, 0x[A-Fa-f0-9]+40\): 0x3 \(mismatch\)$"
            ],
        )
        # Here both start and end are unaligned.
        self.expect(
            "memory tag read {addr}+16 {addr}+80".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+20, 0x[A-Fa-f0-9]+30\): 0x2 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+30, 0x[A-Fa-f0-9]+40\): 0x3 \(mismatch\)\n"
                "\[0x[A-Fa-f0-9]+40, 0x[A-Fa-f0-9]+50\): 0x4 \(mismatch\)$"
            ],
        )

        # For the intial alignment of start/end to granule boundaries the tag manager
        # is used, so this reads 1 tag as it would normally.
        self.expect(
            "memory tag read {addr} {addr}+1".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n" "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0$"
            ],
        )

        # This range is aligned to granules as mte_buf to mte_buf+32 so the result
        # should be 2 granules.
        self.expect(
            "memory tag read {addr} {addr}+17".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)$"
            ],
        )

        # Alignment of this range causes it to become unaligned to 2*granule boundaries.
        self.expect(
            "memory tag read {addr} {addr}+33".format(addr=self.MTE_BUF_ADDR),
            patterns=[
                "Allocation tags:\n"
                "\[0x[A-Fa-f0-9]+00, 0x[A-Fa-f0-9]+10\): 0x0\n"
                "\[0x[A-Fa-f0-9]+10, 0x[A-Fa-f0-9]+20\): 0x1 \(mismatch\)\n",
                "\[0x[A-Fa-f0-9]+20, 0x[A-Fa-f0-9]+30\): 0x2 \(mismatch\)$",
            ],
        )

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_commands_no_mte(self):
        """Test that memory tagging commands fail on an AArch64 corefile without
        any tag segments."""

        self.runCmd("target create --core core.nomte")

        self.expect(
            "memory tag read 0 1",
            substrs=["error: Process does not support memory tagging"],
            error=True,
        )
        # Note that this tells you memory tagging is not supported at all, versus
        # the MTE core file which does support it but does not allow writing tags.
        self.expect(
            "memory tag write 0 1",
            substrs=["error: Process does not support memory tagging"],
            error=True,
        )

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_tag_fault_reason(self):
        """Test that we correctly report the fault reason."""
        self.runCmd("target create --core core.mte")

        # There is no fault address shown here because core files do not include
        # si_addr.
        self.expect(
            "bt",
            substrs=[
                "* thread #1, name = 'a.out.mte', stop reason = signal SIGSEGV: "
                "sync tag check fault"
            ],
        )

    @skipIfLLVMTargetMissing("AArch64")
    def test_mte_ctrl_register(self):
        """Test that we correctly report the mte_ctrl register"."""
        # The register is present even if MTE is not used in the current process
        # and also on targets without MTE, because it controls parts of the
        # overall tagged address ABI as well.
        self.runCmd("target create --core core.nomte")
        self.expect("register read mte_ctrl", substrs=["mte_ctrl = 0x0000000000000000"])

        self.runCmd("target create --core core.mte")
        # The expected value is:
        # * Allowed tags value of 0xFFFF, shifted up by 3 resulting in 0x7fff8.
        # * Bit 1 set to enable synchronous tag faults.
        # * Bit 0 set to enable the tagged address ABI.
        self.expect("register read mte_ctrl", substrs=["mte_ctrl = 0x000000000007fffb"])
