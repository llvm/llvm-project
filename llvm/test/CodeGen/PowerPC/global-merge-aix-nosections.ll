; RUN: rm -rf %t
; RUN: mkdir -p %t

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 -nozero-initialized-in-bss < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:    -nozero-initialized-in-bss --filetype=obj -o %t/global-merge-aix-nosections.o < %s
; RUN: llvm-objdump --syms %t/global-merge-aix-nosections.o | FileCheck %s --check-prefix=OBJ

%struct.Example = type { i32, i8 }

@a = internal global i32 1, align 4
@b = internal global i32 2, align 4
@c = internal global i32 3, align 4
@S1 = internal global %struct.Example { i32 4, i8 5 }, align 4

@y = internal global i32 0, align 4
@z = internal global i32 0, align 4
@l = internal global i32 0, align 4
@u = internal global i16 0, align 2
@S2 = internal global %struct.Example zeroinitializer, align 4

define void @g() {
entry:
  call void @escape(ptr noundef nonnull @a)
  call void @escape(ptr noundef nonnull @b)
  call void @escape(ptr noundef nonnull @c)
  call void @escape(ptr noundef nonnull @S1)
  call void @escape(ptr noundef nonnull @y)
  call void @escape(ptr noundef nonnull @z)
  call void @escape(ptr noundef nonnull @l)
  call void @escape(ptr noundef nonnull @u)
  call void @escape(ptr noundef nonnull @S2)
  ret void
}

declare void @escape(ptr noundef)

; CHECK:        .csect L.._MergedGlobals[RW],2
; CHECK-NEXT:   .lglobl u
; CHECK-NEXT:   .lglobl a
; CHECK-NEXT:   .lglobl b
; CHECK-NEXT:   .lglobl c
; CHECK-NEXT:   .lglobl y
; CHECK-NEXT:   .lglobl z
; CHECK-NEXT:   .lglobl l
; CHECK-NEXT:   .lglobl S1
; CHECK-NEXT:   .lglobl S2
; CHECK-NEXT:   .align  2
; CHECK-NEXT: u:
; CHECK-NEXT:   .vbyte  2, 0
; CHECK-NEXT:   .space  2
; CHECK-NEXT: a:
; CHECK-NEXT:   .vbyte  4, 1
; CHECK-NEXT: b:
; CHECK-NEXT:   .vbyte  4, 2
; CHECK-NEXT: c:
; CHECK-NEXT:   .vbyte  4, 3
; CHECK-NEXT: y:
; CHECK-NEXT:   .vbyte  4, 0
; CHECK-NEXT: z:
; CHECK-NEXT:   .vbyte  4, 0
; CHECK-NEXT: l:
; CHECK-NEXT:   .vbyte  4, 0
; CHECK-NEXT: S1:
; CHECK-NEXT:   .vbyte  4, 4
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .space  3
; CHECK-NEXT: S2:
; CHECK-NEXT:   .space  4
; CHECK-NEXT:   .space  4

; OBJ:      00000000000000ac l     O .data  000000000000002c L.._MergedGlobals
; OBJ-NEXT: 00000000000000ac l     O .data (csect: L.._MergedGlobals)       0000000000000000 u
; OBJ-NEXT: 00000000000000b0 l     O .data (csect: L.._MergedGlobals)       0000000000000000 a
; OBJ-NEXT: 00000000000000b4 l     O .data (csect: L.._MergedGlobals)       0000000000000000 b
; OBJ-NEXT: 00000000000000b8 l     O .data (csect: L.._MergedGlobals)       0000000000000000 c
; OBJ-NEXT: 00000000000000bc l     O .data (csect: L.._MergedGlobals)       0000000000000000 y
; OBJ-NEXT: 00000000000000c0 l     O .data (csect: L.._MergedGlobals)       0000000000000000 z
; OBJ-NEXT: 00000000000000c4 l     O .data (csect: L.._MergedGlobals)       0000000000000000 l
; OBJ-NEXT: 00000000000000c8 l     O .data (csect: L.._MergedGlobals)       0000000000000000 S1
; OBJ-NEXT: 00000000000000d0 l     O .data (csect: L.._MergedGlobals)       0000000000000000 S2
