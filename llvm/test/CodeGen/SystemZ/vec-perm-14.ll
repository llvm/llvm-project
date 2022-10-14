; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
;
; Test that only one vperm of the vector compare is needed for both extracts.

define void @fun() {
; CHECK-LABEL: fun:
; CHECK: vperm
; CHECK-NOT: vperm
bb:
  %tmp = load <4 x i8>, ptr undef
  %tmp1 = icmp eq <4 x i8> zeroinitializer, %tmp
  %tmp2 = extractelement <4 x i1> %tmp1, i32 0
  br i1 %tmp2, label %bb1, label %bb2

bb1:
  unreachable

bb2:
  %tmp3 = extractelement <4 x i1> %tmp1, i32 1
  br i1 %tmp3, label %bb3, label %bb4

bb3:
  unreachable

bb4:
  unreachable
}

; Test that a zero index in the permute vector is used instead of VGBM, with
; a zero index into the other source operand.
define <4 x i8> @fun1(<2 x i8> %arg) {
; CHECK-LABEL:.LCPI1_0:
; CHECK-NEXT:        .byte   1                       # 0x1
; CHECK-NEXT:        .byte   18                      # 0x12
; CHECK-NEXT:        .byte   0                       # 0x0
; CHECK-NEXT:        .byte   18                      # 0x12
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .text
; CHECK-NEXT:        .globl  fun1
; CHECK-NEXT:        .p2align        4
; CHECK-NEXT:        .type   fun1,@function
; CHECK-NEXT: fun1:                                  # @fun1
; CHECK-NEXT:        .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:        larl    %r1, .LCPI1_0
; CHECK-NEXT:        vl      %v0, 0(%r1), 3
; CHECK-NEXT:        vperm   %v24, %v24, %v0, %v0
; CHECK-NEXT:        br      %r14
   %res = shufflevector <2 x i8> %arg, <2 x i8> zeroinitializer,
                        <4 x i32> <i32 1, i32 2, i32 0, i32 3>
   ret <4 x i8> %res
}

; Same, but with the first byte indexing into an element of the zero vector.
define <4 x i8> @fun2(<2 x i8> %arg) {
; CHECK-LABEL:.LCPI2_0:
; CHECK-NEXT:        .byte   0                       # 0x0
; CHECK-NEXT:        .byte   17                      # 0x11
; CHECK-NEXT:        .byte   17                      # 0x11
; CHECK-NEXT:        .byte   0                       # 0x0
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .space  1
; CHECK-NEXT:        .text
; CHECK-NEXT:        .globl  fun2
; CHECK-NEXT:        .p2align        4
; CHECK-NEXT:        .type   fun2,@function
; CHECK-NEXT:fun2:                                   # @fun2
; CHECK-NEXT:        .cfi_startproc
; CHECK-NEXT:# %bb.0:
; CHECK-NEXT:        larl    %r1, .LCPI2_0
; CHECK-NEXT:        vl      %v0, 0(%r1), 3
; CHECK-NEXT:        vperm   %v24, %v0, %v24, %v0
; CHECK-NEXT:        br      %r14
   %res = shufflevector <2 x i8> %arg, <2 x i8> zeroinitializer,
                        <4 x i32> <i32 3, i32 1, i32 1, i32 2>
   ret <4 x i8> %res
}
