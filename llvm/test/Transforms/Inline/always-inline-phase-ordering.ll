; RUN: opt --Os -pass-remarks=inline -S < %s 2>&1 | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64e-apple-macosx13"

; CHECK: remark: <unknown>:0:0: 'wibble' inlined into 'bar.8' with (cost=always): always inline attribute
; CHECK: remark: <unknown>:0:0: 'wibble' inlined into 'pluto' with (cost=always): always inline attribute
; CHECK: remark: <unknown>:0:0: 'snork' inlined into 'blam' with (cost=always): always inline attribute
; CHECK: remark: <unknown>:0:0: 'wobble' inlined into 'blam' with (cost=always): always inline attribute
; CHECK: remark: <unknown>:0:0: 'spam' inlined into 'blam' with (cost=65, threshold=75)
; CHECK: remark: <unknown>:0:0: 'wibble.1' inlined into 'widget' with (cost=30, threshold=75)
; CHECK: remark: <unknown>:0:0: 'widget' inlined into 'bar.8' with (cost=30, threshold=75)
; CHECK: remark: <unknown>:0:0: 'barney' inlined into 'wombat' with (cost=30, threshold=75)

define linkonce_odr void @wombat(ptr %arg) #0 {
bb:
  call void @barney()
  ret void
}

define i1 @foo() {
bb:
  call void @wombat(ptr null)
  unreachable
}

define linkonce_odr void @pluto() #1 !prof !38 {
bb:
  call void @wibble()
  ret void
}

; Function Attrs: alwaysinline
define linkonce_odr void @wibble() #2 {
bb:
  call void @widget()
  ret void
}

define linkonce_odr void @widget() {
bb:
  call void @wibble.1()
  ret void
}

define linkonce_odr void @wibble.1() {
bb:
  %0 = call i32 @foo.2()
  call void @blam()
  ret void
}

declare i32 @foo.2()

define linkonce_odr void @blam() {
bb:
  %tmp = call i32 @snork()
  %tmpv1 = call ptr @wombat.3()
  call void @eggs()
  %tmpv2 = call ptr @wombat.3()
  ret void
}

; Function Attrs: alwaysinline
define linkonce_odr i32 @snork() #2 {
bb:
  %tmpv1 = call i32 @spam()
  %tmpv2 = call i32 @wobble()
  call void @widget.4(i32 %tmpv2)
  ret i32 0
}

declare void @eggs()

declare ptr @wombat.3()

define linkonce_odr i32 @spam() {
bb:
  %tmpv1 = call i32 @wombat.6()
  %tmpv2 = call i64 @wobble.5(i8 0)
  %tmpv3 = call i64 @bar()
  ret i32 0
}

; Function Attrs: alwaysinline
define linkonce_odr i32 @wobble() #2 {
bb:
  %tmpv = call i64 @wobble.5(i8 0)
  %tmpv1 = call i64 @eggs.7()
  %tmpv2 = call i64 @wobble.5(i8 0)
  %tmpv3 = call i64 @eggs.7()
  %tmpv4 = lshr i64 %tmpv1, 1
  %tmpv5 = trunc i64 %tmpv4 to i32
  %tmpv6 = xor i32 %tmpv5, 23
  ret i32 %tmpv6
}

declare void @widget.4(i32)

declare i64 @bar()

declare i64 @wobble.5(i8)

declare i32 @wombat.6()

declare i64 @eggs.7()

define linkonce_odr void @barney() {
bb:
  call void @bar.8()
  call void @pluto()
  unreachable
}

define linkonce_odr void @bar.8() {
bb:
  call void @wibble()
  ret void
}

attributes #0 = { "frame-pointer"="non-leaf" }
attributes #1 = { "target-cpu"="apple-m1" }
attributes #2 = { alwaysinline }

!llvm.module.flags = !{!0, !1, !30, !31, !32, !36, !37}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 13, i32 3]}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 864540306756}
!5 = !{!"MaxCount", i64 6596759955}
!6 = !{!"MaxInternalCount", i64 2828618424}
!7 = !{!"MaxFunctionCount", i64 6596759955}
!8 = !{!"NumCounts", i64 268920}
!9 = !{!"NumFunctions", i64 106162}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 5109654023, i32 2}
!15 = !{i32 100000, i64 2480859832, i32 25}
!16 = !{i32 200000, i64 1566552109, i32 70}
!17 = !{i32 300000, i64 973667919, i32 140}
!18 = !{i32 400000, i64 552159773, i32 263}
!19 = !{i32 500000, i64 353879860, i32 463}
!20 = !{i32 600000, i64 187122455, i32 799}
!21 = !{i32 700000, i64 105465980, i32 1419}
!22 = !{i32 800000, i64 49243829, i32 2620}
!23 = !{i32 900000, i64 15198227, i32 5898}
!24 = !{i32 950000, i64 5545670, i32 10696}
!25 = !{i32 990000, i64 804816, i32 25738}
!26 = !{i32 999000, i64 73999, i32 53382}
!27 = !{i32 999900, i64 6530, i32 83503}
!28 = !{i32 999990, i64 899, i32 110416}
!29 = !{i32 999999, i64 120, i32 130201}
!30 = !{i32 7, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 1, !"wchar_size", i32 4}
!34 = !{!35}
!35 = !{i32 0, i1 false}
!36 = !{i32 8, !"PIC Level", i32 2}
!37 = !{i32 7, !"frame-pointer", i32 1}
!38 = !{!"function_entry_count", i64 15128150}
