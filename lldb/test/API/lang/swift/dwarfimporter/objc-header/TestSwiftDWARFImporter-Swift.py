import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftDWARFImporter_Swift(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def build(self):
        include = self.getBuildArtifact('include')
        inputs = self.getSourcePath(os.path.join('Inputs', 'Modules'))
        lldbutil.mkdir_p(include)
        import shutil
        for f in ['module.modulemap', 'objc-header.h']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(include, f))

        super(TestSwiftDWARFImporter_Swift, self).build()

        # Remove the header files to thwart ClangImporter.
        self.assertTrue(os.path.isdir(include))
        shutil.rmtree(include)

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        dylib = self.getBuildArtifact('libLibrary.dylib')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images = [dylib])

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        self.expect("target var -d run myobj", substrs=["(ObjCClass)"])
