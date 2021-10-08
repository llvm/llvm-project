import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


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
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
        self.build()
        dylib = self.getBuildArtifact('libLibrary.dylib')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images = [dylib])

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        self.expect("target var -d run myobj", substrs=["(ObjCClass)"])

        found = 0
        response = 0
        logfile = open(log, "r")
        for line in logfile:
            if 'SwiftDWARFImporterDelegate::lookupValue("ObjCClass")' in line:
                found += 1
            elif found == 1 and response == 0 and 'SwiftDWARFImporterDelegate' in line:
                self.assertTrue('from debug info' in line, line)
                response += 1
            elif found == 2 and response == 1 and 'SwiftDWARFImporterDelegate' in line:
                self.assertTrue('types collected' in line, line)
                response += 1
        self.assertEqual(found, 1)
        self.assertEqual(response, 1)
