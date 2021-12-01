import bz2
import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfFBSDVMCoreSupportMissing
class FreeBSDKernelVMCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def make_target(self, src_filename):
        src = self.getSourcePath(src_filename)
        dest = self.getBuildArtifact("kernel")
        self.yaml2obj(src, dest, max_size=30*1024*1024)
        return self.dbg.CreateTarget(dest)

    def make_vmcore(self, src_filename):
        src = self.getSourcePath(src_filename)
        dest = self.getBuildArtifact("vmcore")
        with bz2.open(src, "rb") as inf:
            with open(dest, "wb") as outf:
                shutil.copyfileobj(inf, outf)
        return dest

    def do_test(self, kernel_yaml, vmcore_bz2, bt_expected, regs_expected,
                hz_value=100):
        target = self.make_target(kernel_yaml)
        vmcore_file = self.make_vmcore(vmcore_bz2)
        process = target.LoadCore(vmcore_file)

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), 0)

        # test memory reading
        self.expect("expr -- *(int *) &hz",
                    substrs=["(int) $0 = %d" % hz_value])

        main_mod = target.GetModuleAtIndex(0)
        hz_addr = (main_mod.FindSymbols("hz")[0].symbol.addr
                   .GetLoadAddress(target))
        error = lldb.SBError()
        self.assertEqual(process.ReadMemory(hz_addr, 4, error),
                         struct.pack("<I", hz_value))

        # test backtrace
        self.assertEqual(
            [process.GetThreadAtIndex(0).GetFrameAtIndex(i).addr
             .GetLoadAddress(target) for i in range(len(bt_expected))],
            bt_expected)

        # test registers
        regs = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        reg_values = {}
        for regset in regs:
            for reg in regset:
                if reg.value is None:
                    continue
                reg_values[reg.name] = reg.value
        self.assertEqual(reg_values, regs_expected)

        self.dbg.DeleteTarget(target)

    def test_amd64_full_vmcore(self):
        self.do_test("kernel-amd64.yaml", "vmcore-amd64-full.bz2",
                     [0xffffffff80c09ade, 0xffffffff80c09916,
                      0xffffffff80c09d90, 0xffffffff80c09b93,
                      0xffffffff80c57d91, 0xffffffff80c19e71,
                      0xffffffff80c192bc, 0xffffffff80c19933,
                      0xffffffff80c1977f, 0xffffffff8108ba8c,
                      0xffffffff810620ce],
                     {"rbx": "0x0000000000000000",
                      "rbp": "0xfffffe0085cb2760",
                      "rsp": "0xfffffe0085cb2748",
                      "r12": "0xfffffe0045a6c300",
                      "r13": "0xfffff800033693a8",
                      "r14": "0x0000000000000000",
                      "r15": "0xfffff80003369380",
                      "rip": "0xffffffff80c09ade",
                      })

    def test_amd64_minidump(self):
        self.do_test("kernel-amd64.yaml", "vmcore-amd64-minidump.bz2",
                     [0xffffffff80c09ade, 0xffffffff80c09916,
                      0xffffffff80c09d90, 0xffffffff80c09b93,
                      0xffffffff80c57d91, 0xffffffff80c19e71,
                      0xffffffff80c192bc, 0xffffffff80c19933,
                      0xffffffff80c1977f, 0xffffffff8108ba8c,
                      0xffffffff810620ce],
                     {"rbx": "0x0000000000000000",
                      "rbp": "0xfffffe00798c4760",
                      "rsp": "0xfffffe00798c4748",
                      "r12": "0xfffffe0045b11c00",
                      "r13": "0xfffff800033693a8",
                      "r14": "0x0000000000000000",
                      "r15": "0xfffff80003369380",
                      "rip": "0xffffffff80c09ade",
                      })

    def test_arm64_minidump(self):
        self.do_test("kernel-arm64.yaml", "vmcore-arm64-minidump.bz2",
                     [0xffff0000004b6e78],  # TODO: fix unwinding
                      {"x0": "0x0000000000000000",
                       "x1": "0x0000000000000000",
                       "x2": "0x0000000000000000",
                       "x3": "0x0000000000000000",
                       "x4": "0x0000000000000000",
                       "x5": "0x0000000000000000",
                       "x6": "0x0000000000000000",
                       "x7": "0x0000000000000000",
                       "x8": "0xffffa00001548700",
                       "x9": "0x0000000000000000",
                       "x10": "0xffffa00000e04580",
                       "x11": "0x0000000000000000",
                       "x12": "0x000000000008950a",
                       "x13": "0x0000000000089500",
                       "x14": "0x0000000000000039",
                       "x15": "0x0000000000000000",
                       "x16": "0x00000000ffffffd8",
                       "x17": "0x0000000000000000",
                       "x18": "0xffff000000e6d380",
                       "x19": "0xffff000000af9000",
                       "x20": "0xffff000000b82000",
                       "x21": "0xffffa00000319da8",
                       "x22": "0xffff000000b84000",
                       "x23": "0xffff000000b84000",
                       "x24": "0xffff000000b55000",
                       "x25": "0x0000000000000000",
                       "x26": "0x0000000000040800",
                       "x27": "0x0000000000000000",
                       "x28": "0x00000000002019ca",
                       "fp": "0xffff0000d58f23b0",
                       "sp": "0xffff0000d58f23b0",
                       "pc": "0xffff0000004b6e78",
                       },
                     hz_value=1000)

    def test_i386_minidump(self):
        self.do_test("kernel-i386.yaml", "vmcore-i386-minidump.bz2",
                     [0x010025c5, 0x01002410, 0x010027d5, 0x01002644,
                      0x01049a2f, 0x01011077, 0x01010780, 0x01010c7a,
                      0x01010ab2, 0x013e9e2d, 0xffc033f9],
                     {"ebp": "0x151968e4",
                      "esp": "0x151968d8",
                      "esi": "0x0c77aa80",
                      "edi": "0x03f0dc80",
                      "eip": "0x010025c5",
                      })
