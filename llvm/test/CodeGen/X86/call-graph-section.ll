;; Tests that we store the type identifiers in .callgraph section of the binary.

; RUN: llc --call-graph-section -filetype=obj -o - < %s | \
; RUN: llvm-readelf -x .callgraph - | FileCheck %s

declare !type !4 void @foo() #0

declare !type !5 noundef i32 @bar(i8 signext %a) #0

declare !type !6 noundef ptr @baz(ptr %a) #0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @main() #0 !type !7 {
entry:
  %retval = alloca i32, align 4
  %fp_foo = alloca ptr, align 8
  %a = alloca i8, align 1
  %fp_bar = alloca ptr, align 8
  %fp_baz = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp_foo, align 8
  %0 = load ptr, ptr %fp_foo, align 8
  call void (...) %0() [ "callee_type"(metadata !"_ZTSFvE.generalized") ]
  store ptr @bar, ptr %fp_bar, align 8
  %1 = load ptr, ptr %fp_bar, align 8
  %2 = load i8, ptr %a, align 1
  %call = call i32 %1(i8 signext %2) [ "callee_type"(metadata !"_ZTSFicE.generalized") ]
  store ptr @baz, ptr %fp_baz, align 8
  %3 = load ptr, ptr %fp_baz, align 8
  %call1 = call ptr %3(ptr %a) [ "callee_type"(metadata !"_ZTSFPvS_E.generalized") ]
  call void @foo() [ "callee_type"(metadata !"_ZTSFvE.generalized") ]
  %4 = load i8, ptr %a, align 1
  %call2 = call i32 @bar(i8 signext %4) [ "callee_type"(metadata !"_ZTSFicE.generalized") ]
  %call3 = call ptr @baz(ptr %a) [ "callee_type"(metadata !"_ZTSFPvS_E.generalized") ]
  ret void
}

;; Check that the numeric type id (md5 hash) for the below type ids are emitted
;; to the callgraph section.

; CHECK: Hex dump of section '.callgraph':

; CHECK-DAG: 2444f731 f5eecb3e
!4 = !{i64 0, !"_ZTSFvE.generalized"}
; CHECK-DAG: 5486bc59 814b8e30
!5 = !{i64 0, !"_ZTSFicE.generalized"}
; CHECK-DAG: 7ade6814 f897fd77
!6 = !{i64 0, !"_ZTSFPvS_E.generalized"}
; CHECK-DAG: caaf769a 600968fa
!7 = !{i64 0, !"_ZTSFiE.generalized"}
