; RUN: rm -rf %t
; RUN: mkdir -p %t

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 --filetype=obj -o %t/global-merge-aix-sections.o < %s
; RUN; llvm-objdump --syms %t/global-merge-aix-sections.o | FileCheck %s --check-prefix=DATA

%struct.Example = type { i32, i8 }

@y = internal global i32 0, section "mycsect", align 4
@z = internal global i32 0, section "mycsect", align 4
@l = internal global i32 0, align 4
@u = internal global i16 0, section "mycsect", align 2
@myStruct1 = internal global %struct.Example zeroinitializer, section "mycsect", align 4

; Function Attrs: nounwind
define void @g() {
entry:
  tail call void @f(ptr noundef nonnull @y, ptr noundef nonnull @z)
  tail call void @f(ptr noundef nonnull @l, ptr noundef nonnull @z)
  tail call void @h(ptr noundef nonnull @u)
  tail call void @s(ptr noundef nonnull @myStruct1)
  ret void
}

declare void @f(ptr noundef, ptr noundef)
declare void @h(ptr noundef)
declare void @s(ptr noundef)

; CHECK: .csect mycsect[RW],2
; CHECK-NEXT: .lglobl u # @_MergedGlobals
; CHECK-NEXT: .lglobl y
; CHECK-NEXT: .lglobl z
; CHECK-NEXT: .lglobl myStruct1
; CHECK-NEXT: .align  2
; CHECK-NEXT: L.._MergedGlobals:
; CHECK-NEXT: u:
; CHECK-NEXT:        .space  2
; CHECK-NEXT:        .space  2
; CHECK-NEXT: y:
; CHECK-NEXT:        .space  4
; CHECK-NEXT: z:
; CHECK-NEXT:        .space  4
; CHECK-NEXT: myStruct1:
; CHECK-NEXT:        .space  8

; DATA: 00000078 l     O .data  00000014 mycsect
; DATA-NEXT: 00000078 l     O .data (csect: mycsect)         00000000 u
; DATA-NEXT: 0000007c l     O .data (csect: mycsect)         00000000 y
; DATA-NEXT: 00000080 l     O .data (csect: mycsect)         00000000 z
; DATA-NEXT: 00000084 l     O .data (csect: mycsect)         00000000 myStruct1
