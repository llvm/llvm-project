## Check interpretation of the packed unwind info format.

// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func1
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 88
// CHECK-NEXT:     RegF: 7
// CHECK-NEXT:     RegI: 10
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 160
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       stp d14, d15, [sp, #128]
// CHECK-NEXT:       stp d12, d13, [sp, #112]
// CHECK-NEXT:       stp d10, d11, [sp, #96]
// CHECK-NEXT:       stp d8, d9, [sp, #80]
// CHECK-NEXT:       stp x27, x28, [sp, #64]
// CHECK-NEXT:       stp x25, x26, [sp, #48]
// CHECK-NEXT:       stp x23, x24, [sp, #32]
// CHECK-NEXT:       stp x21, x22, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-144]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func2
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 48
// CHECK-NEXT:     RegF: 2
// CHECK-NEXT:     RegI: 3
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       str d10, [sp, #40]
// CHECK-NEXT:       stp d8, d9, [sp, #24]
// CHECK-NEXT:       str x21, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-48]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func3
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 40
// CHECK-NEXT:     RegF: 3
// CHECK-NEXT:     RegI: 1
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       stp d10, d11, [sp, #24]
// CHECK-NEXT:       stp d8, d9, [sp, #8]
// CHECK-NEXT:       str x19, [sp, #-48]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func4
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 1
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #32
// CHECK-NEXT:       stp d8, d9, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func5
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 56
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 1
// CHECK-NEXT:     HomedParameters: Yes
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #32
// CHECK-NEXT:       stp x6, x7, [sp, #56]
// CHECK-NEXT:       stp x4, x5, [sp, #40]
// CHECK-NEXT:       stp x2, x3, [sp, #24]
// CHECK-NEXT:       stp x0, x1, [sp, #8]
// CHECK-NEXT:       str x19, [sp, #-80]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func6
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 48
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: Yes
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #48
// CHECK-NEXT:       stp x6, x7, [sp, #48]
// CHECK-NEXT:       stp x4, x5, [sp, #32]
// CHECK-NEXT:       stp x2, x3, [sp, #16]
// CHECK-NEXT:       stp x0, x1, [sp, #-64]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func7
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       str lr, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func8
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 1
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       INVALID!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func9
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       str lr, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func10
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 3
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       stp x21, lr, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func11
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func12
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 40
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 544
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #0]
// CHECK-NEXT:       sub sp, sp, #528
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func13
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 48
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 4112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #0]
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       sub sp, sp, #4080
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func14
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 4112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       sub sp, sp, #4080
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func15
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 560
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #544
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func16
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 56
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: Yes
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #32
// CHECK-NEXT:       stp x6, x7, [sp, #56]
// CHECK-NEXT:       stp x4, x5, [sp, #40]
// CHECK-NEXT:       stp x2, x3, [sp, #24]
// CHECK-NEXT:       stp x0, x1, [sp, #8]
// CHECK-NEXT:       str lr, [sp, #-80]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .text
        .globl func1
func1:
func2:
func3:
func4:
func5:
func6:
func7:
func8:
func9:
func10:
func11:
func12:
func13:
func14:
func15:
func16:
        ret

        .section .pdata,"dr"
        .long func1@IMGREL
        .long 0x050ae059 // FunctionLength=22 RegF=7 RegI=10 H=0 CR=0 FrameSize=10
        .long func2@IMGREL
        .long 0x01834031 // FunctionLength=12 RegF=2 RegI=3 H=0 CR=0 FrameSize=3
        .long func3@IMGREL
        .long 0x01816029 // FunctionLength=10 RegF=3 RegI=1 H=0 CR=0 FrameSize=3
        .long func4@IMGREL
        .long 0x01802019 // FunctionLength=6  RegF=1 RegI=0 H=0 CR=0 FrameSize=3
        .long func5@IMGREL
        .long 0x03910039 // FunctionLength=14 RegF=0 RegI=1 H=1 CR=0 FrameSize=7
        .long func6@IMGREL
        .long 0x03900031 // FunctionLength=12 RegF=0 RegI=0 H=1 CR=0 FrameSize=7
        .long func7@IMGREL
        .long 0x01200019 // FunctionLength=6  RegF=0 RegI=0 H=0 CR=1 FrameSize=2
        .long func8@IMGREL
        .long 0x01210019 // FunctionLength=6  RegF=0 RegI=1 H=0 CR=1 FrameSize=2
        .long func9@IMGREL
        .long 0x01220021 // FunctionLength=8  RegF=0 RegI=2 H=0 CR=1 FrameSize=2
        .long func10@IMGREL
        .long 0x01a30021 // FunctionLength=8  RegF=0 RegI=3 H=0 CR=1 FrameSize=3
        .long func11@IMGREL
        .long 0x01e20021 // FunctionLength=8  RegF=0 RegI=2 H=0 CR=3 FrameSize=3
        .long func12@IMGREL
        .long 0x11620029 // FunctionLength=10 RegF=0 RegI=2 H=0 CR=3 FrameSize=34
        .long func13@IMGREL
        .long 0x80e20031 // FunctionLength=12 RegF=0 RegI=2 H=0 CR=3 FrameSize=257
        .long func14@IMGREL
        .long 0x80820021 // FunctionLength=8  RegF=0 RegI=2 H=0 CR=0 FrameSize=257
        .long func15@IMGREL
        .long 0x11820019 // FunctionLength=6  RegF=0 RegI=2 H=0 CR=0 FrameSize=34
        .long func16@IMGREL
        .long 0x03b00039 // FunctionLength=14 RegF=0 RegI=0 H=1 CR=1 FrameSize=7
