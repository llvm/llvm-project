; Combining the transfers into register pairs would leave nothing to pull up.
; RUN: llc -march=hexagon < %s -hexagon-expand-condsets=false -disable-merge-into-combines \
; RUN:     -prevent-duplex-sep-v2=false -mno-pairing -mno-compound -rdf-opt=0 \
; RUN:     -hexagon-initial-cfg-cleanup=0 -hexagon-instsimplify=0 \
; RUN:     | FileCheck %s
; RUN: llc -march=hexagon < %s -hexagon-expand-condsets=false -disable-merge-into-combines -mno-pairing \
; RUN:     -mno-compound -rdf-opt=0 -hexagon-initial-cfg-cleanup=0 \
; RUN:     -hexagon-instsimplify=0 \
; RUN:     | FileCheck --check-prefix=CHECK1 %s

; CHECK: if (p{{[0-3]}}) r{{[0-9]*}} = add(r{{[0-9]*}},#0)
; CHECK: if (p{{[0-3]}}) r{{[0-9]*}} = ##baar1
; CHECK: r{{[0-9]*}} = #0
; CHECK: r{{[0-9]*}} = mux({{.*}}##baar)
; CHECK: r{{[0-9]*}} = #0

; CHECK1: r{{[0-9]*}} = r{{[0-9]*}}
; CHECK1: r{{[0-9]*}} = mux({{.*}}##baar)
; CHECK1: r{{[0-9]*}} = memw(r{{[0-9]*}}+#4)
; CHECK1-DAG: r{{[0-9]*}} = #0
; CHECK1-DAG: r{{[0-9]*}} = memw
; CHECK1-DAG: p{{[0-9]*}} = r{{[0-9]*}}

target triple = "hexagon"

%0 = type { %1 }
%1 = type { i32, ptr }
%2 = type { %3, %4, i8, %8, %24, %29, i8, %30, %30, %36, %37, i32, %38, %58 }
%3 = type { i16, [2 x i8] }
%4 = type { i8, %5 }
%5 = type { i32, %6 }
%6 = type { ptr }
%7 = type { i32, [5 x i8] }
%8 = type { %9, %10, %13 }
%9 = type { i8, [3 x i8] }
%10 = type { %11, i8, i8, %12 }
%11 = type { i8, [3 x i8] }
%12 = type { i32, ptr }
%13 = type { %14, %15, %18, i8, %23 }
%14 = type { i8, [3 x i8] }
%15 = type { i32, ptr }
%16 = type { %17, i32, i8, i8, i8, %58 }
%17 = type { i8, [3 x i8] }
%18 = type { i32, ptr }
%19 = type { i32, %20 }
%20 = type { i32, ptr }
%21 = type { i16, %22 }
%22 = type { i32, ptr }
%23 = type { i32, i32 }
%24 = type { %25, %26, %28, %13 }
%25 = type { i8, [3 x i8] }
%26 = type { %27, %27, i8, i8, i8, i8, i8, i8, i8, %27, %27, %27, %27 }
%27 = type { i32, [4 x i8] }
%28 = type { i32, [6 x i8] }
%29 = type { i32, ptr }
%30 = type { %31 }
%31 = type { i32, ptr }
%32 = type { %33 }
%33 = type { i32, ptr }
%34 = type { %35 }
%35 = type { i32, ptr }
%36 = type { i32 }
%37 = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%38 = type { i32, ptr }
%39 = type { i8, %40 }
%40 = type { i32, %41 }
%41 = type { ptr }
%42 = type { %43, %44, i8, %46, %55, %58 }
%43 = type { i8, [3 x i8] }
%44 = type { i32, %45 }
%45 = type { ptr }
%46 = type { %47, %10, %48 }
%47 = type { i8, [3 x i8] }
%48 = type { %49, %15, %50, i8, %23 }
%49 = type { i8, [3 x i8] }
%50 = type { i32, ptr }
%51 = type { i32, %52 }
%52 = type { i32, ptr }
%53 = type { i16, %54 }
%54 = type { i32, ptr }
%55 = type { %56, %26, %36, %57, %48, %37, i8, i32 }
%56 = type { i8, [3 x i8] }
%57 = type { i32, [6 x i8] }
%58 = type { i32, ptr, ptr }
%59 = type { ptr, ptr, ptr }
%60 = type { i16, i16, i16 }

@g0 = external constant %0, align 4
@g1 = external constant %0, align 4

; Function Attrs: optsize
declare zeroext i8 @foo1() #0

; Function Attrs: optsize
declare void @bar(ptr, i32) #0

; Function Attrs: optsize
declare zeroext i8 @baar(i16 signext) #0

; Function Attrs: optsize
declare zeroext i8 @baar1(i16 signext) #0

; Function Attrs: nounwind optsize ssp
define i32 @fun1(ptr readonly, ptr, ptr nocapture, i8 zeroext) #1 {
  %5 = tail call zeroext i8 @foo1() #2
  br i1 poison, label %7, label %6, !prof !0

; <label>:6                                       ; preds = %4
  br i1 poison, label %7, label %8, !prof !0

; <label>:7                                       ; preds = %6, %4
  unreachable

; <label>:8                                       ; preds = %6
  %9 = icmp ne i8 %3, 0
  %10 = icmp ult i8 %5, 32
  %11 = or i1 %9, %10
  %12 = select i1 %11, i32 32, i32 40
  br i1 false, label %13, label %14

; <label>:13                                      ; preds = %8
  unreachable

; <label>:14                                      ; preds = %8
  %15 = select i1 %9, ptr @baar, ptr @baar1
  %16 = select i1 %9, ptr @g0, ptr @g1
  %17 = tail call zeroext i8 %15(i16 signext 0) #2
  %18 = icmp eq i8 %17, 0
  br i1 %18, label %20, label %19

; <label>:19                                      ; preds = %14
  br label %21

; <label>:20                                      ; preds = %14
  tail call void @bar(ptr %16, i32 0) #2
  unreachable

; <label>:21                                      ; preds = %37, %19
  %22 = phi i32 [ 0, %19 ], [ %38, %37 ]
  br i1 false, label %26, label %23

; <label>:23                                      ; preds = %21
  br i1 poison, label %37, label %24

; <label>:24                                      ; preds = %23
  %25 = getelementptr inbounds %21, ptr poison, i32 %22, i32 1, i32 1
  br label %28

; <label>:26                                      ; preds = %21
  unreachable

; <label>:27                                      ; preds = %34
  br i1 poison, label %28, label %37

; <label>:28                                      ; preds = %27, %24
  br i1 false, label %33, label %29

; <label>:29                                      ; preds = %32, %28
  br i1 false, label %30, label %32

; <label>:30                                      ; preds = %29
  br i1 poison, label %31, label %32

; <label>:31                                      ; preds = %30
  br i1 false, label %34, label %32

; <label>:32                                      ; preds = %31, %30, %29
  br i1 false, label %29, label %33

; <label>:33                                      ; preds = %32, %28
  br label %34

; <label>:34                                      ; preds = %33, %31
  %35 = icmp eq i32 0, %12
  br i1 %35, label %36, label %27

; <label>:36                                      ; preds = %34
  unreachable

; <label>:37                                      ; preds = %27, %23
  %38 = add nuw i32 %22, 1
  br label %21
}

attributes #0 = { optsize }
attributes #1 = { nounwind optsize ssp }
attributes #2 = { nounwind optsize }

!0 = !{!"branch_weights", i32 4, i32 64}
