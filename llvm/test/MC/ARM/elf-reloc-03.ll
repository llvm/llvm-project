;; RUN: llc -mtriple=armv7-linux-gnueabi -O3  \
;; RUN:    -mcpu=cortex-a8 -mattr=-neon -mattr=+vfp2  -arm-reserve-r9  \
;; RUN:    -filetype=obj %s -o - | \
;; RUN:   elf-dump --dump-section-data | FileCheck -check-prefix=OBJ %s

;; FIXME: This file needs to be in .s form!
;; The args to llc are there to constrain the codegen only.
;; 
;; Ensure no regression on ARM/gcc compatibility for 
;; emitting explicit symbol relocs for nonexternal symbols 
;; versus section symbol relocs (with offset) - 
;;
;; Default llvm behavior is to emit as section symbol relocs nearly
;; everything that is not an undefined external. Unfortunately, this 
;; diverges from what codesourcery ARM/gcc does!
;;
;; Verifies that internal constants appear as explict symbol relocs


target triple = "armv7-none-linux-gnueabi"

@startval = global i32 5
@vtable = internal constant [10 x i32 (...)*] [i32 (...)* bitcast (i32 ()* @foo0 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo1 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo2 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo3 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo4 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo5 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo6 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo7 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo8 to i32 (...)*), i32 (...)* bitcast (i32 ()* @foo9 to i32 (...)*)]

declare i32 @mystrlen(i8* nocapture %s) nounwind readonly 

declare void @myhextochar(i32 %n, i8* nocapture %buffer) nounwind 

define internal i32 @foo0() nounwind readnone {
entry:
  ret i32 0
}

define internal i32 @foo1() nounwind readnone {
entry:
  ret i32 1
}

define internal i32 @foo2() nounwind readnone {
entry:
  ret i32 2
}

define internal i32 @foo3() nounwind readnone {
entry:
  ret i32 3
}

define internal i32 @foo4() nounwind readnone {
entry:
  ret i32 4
}

define internal i32 @foo5() nounwind readnone {
entry:
  ret i32 55
}

define internal i32 @foo6() nounwind readnone {
entry:
  ret i32 6
}

define internal i32 @foo7() nounwind readnone {
entry:
  ret i32 7
}

define internal i32 @foo8() nounwind readnone {
entry:
  ret i32 8
}

define internal i32 @foo9() nounwind readnone {
entry:
  ret i32 9
}

define i32 @main() nounwind {
entry:
  %0 = load i32* @startval, align 4
  %1 = getelementptr inbounds [10 x i32 (...)*]* @vtable, i32 0, i32 %0
  %2 = load i32 (...)** %1, align 4
  %3 = tail call i32 (...)* %2() nounwind
  tail call void @exit(i32 %3) noreturn nounwind
  unreachable
}

declare void @exit(i32) noreturn nounwind

;; OBJ:           Relocation 1
;; OBJ-NEXT:     'r_offset', 
;; OBJ-NEXT:     'r_sym', 0x00000c
;; OBJ-NEXT:     'r_type', 0x2b

;; OBJ:      Symbol 12
;; OBJ-NEXT:    'vtable'
