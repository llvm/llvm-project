"""
Test Intel(R) MPX registers do not get overwritten by AVX data.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MPXOffsetIntersectionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    AVX_REGS = ('ymm' + str(i) for i in range(16))
    YMM_VALUE = '{' + ' '.join(('0x00' for _ in range(32))) + '}'

    MPX_REGULAR_REGS = ('bnd0', 'bnd1', 'bnd2', 'bnd3')
    MPX_CONFIG_REGS = ('bndcfgu', 'bndstatus')
    BND_VALUE = '{' + ' '.join(('0xff' for _ in range(16))) + '}'

    @skipIf(oslist=no_match(['linux']))
    @skipIf(archs=no_match(['x86_64']))
    def test_mpx_registers_offset_intersection(self):
        """Test if AVX data does not overwrite MPX values."""
        self.build()
        self.mpx_registers_offset_intersection()

    def mpx_registers_offset_intersection(self):
        exe = self.getBuildArtifact('a.out')
        self.runCmd('file ' + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd('run', RUN_SUCCEEDED)
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)
        currentFrame = thread.GetFrameAtIndex(0)

        has_avx = False
        has_mpx = False
        for registerSet in currentFrame.GetRegisters():
            if 'advanced vector extensions' in registerSet.GetName().lower():
                has_avx = True
            if 'memory protection extension' in registerSet.GetName().lower():
                has_mpx = True
        if not (has_avx and has_mpx):
            self.skipTest('Both AVX and MPX registers must be supported.')

        for reg in self.AVX_REGS:
            self.runCmd('register write ' + reg + " '" + self.YMM_VALUE + " '")
        for reg in self.MPX_REGULAR_REGS + self.MPX_CONFIG_REGS:
            self.runCmd('register write ' + reg + " '" + self.BND_VALUE + " '")

        self.verify_mpx()
        self.verify_avx()
        self.verify_mpx()

    def verify_mpx(self):
        for reg in self.MPX_REGULAR_REGS:
            self.expect('register read ' + reg,
                        substrs = [reg + ' = {0xffffffffffffffff 0xffffffffffffffff}'])
        for reg in self.MPX_CONFIG_REGS:
            self.expect('register read ' + reg,
                        substrs = [reg + ' = {0xff 0xff 0xff 0xff 0xff 0xff 0xff 0xff}'])

    def verify_avx(self):
        for reg in self.AVX_REGS:
            self.expect('register read ' + reg, substrs = [reg + ' = ' + self.YMM_VALUE])
