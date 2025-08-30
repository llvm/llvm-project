; RUN: rm -rf %t
; RUN: split-file %s  %t
; RUN: llvm-as %t/f1.ll -o %t/f1.bc
; RUN: llvm-as %t/f2.ll -o %t/f2.bc
; RUN: llvm-lto  -filetype=asm  -function-sections=1 %t/f1.bc %t/f2.bc -o %t/fc.s
; RUN: cat %t/fc.s | FileCheck %s

;; Test that the renaming mechanism for the profiling counters and data
;; symbols section works when performing an LTO link with symbols with
;; name clashes from different modules.


;--- f1.ll
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__llvm_profile_raw_version = weak hidden constant i64 72057594037927944
@__profc_func1 = private global [2 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_func1 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 -2545542355363006406, i64 146835647075900052, i32 sub (i32 ptrtoint (ptr @__profc_func1 to i32), i32 ptrtoint (ptr @__profd_func1 to i32)), ptr @func1.local, ptr null, i32 2, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !0
@__profc_file1.c_internal_func = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_file1.c_internal_func = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 2905054957054668920, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_file1.c_internal_func to i32), i32 ptrtoint(ptr @__profd_file1.c_internal_func to i32)), ptr @internal_func, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !1
@__profc_escape1 = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_escape1 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 3473293639883741762, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_escape1 to i32), i32 ptrtoint (ptr @__profd_escape1 to i32)), ptr @escape1.local, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !2
@__llvm_prf_nm = private constant [37 x i8] c"#\00func1\01file1.c:internal_func\01escape1", section "__llvm_prf_names", align 1
@llvm.used = appending global [4 x ptr] [ptr @__profd_func1, ptr @__profd_file1.c_internal_func, ptr @__profd_escape1, ptr @__llvm_prf_nm], section "llvm.metadata"
@__llvm_profile_filename = weak hidden constant [19 x i8] c"default_%m.profraw\00"

@func1.local = private alias i64 (i32), ptr @func1
@escape1.local = private alias ptr (), ptr @escape1

; Function Attrs: noinline nounwind optnone
define i64 @func1(i32 noundef %n) #0 {
entry:
  %retval = alloca i64, align 8
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %pgocount = load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc_func1, i32 0, i32 1), align 8
  %1 = add i64 %pgocount, 1
  store i64 %1, ptr getelementptr inbounds ([2 x i64], ptr @__profc_func1, i32 0, i32 1), align 8
  store i64 0, ptr %retval, align 8
  br label %return

if.end:                                           ; preds = %entry
  %pgocount1 = load i64, ptr @__profc_func1, align 8
  %2 = add i64 %pgocount1, 1
  store i64 %2, ptr @__profc_func1, align 8
  %3 = load i32, ptr %n.addr, align 4
  %call = call i64 @internal_func(i32 noundef %3)
  store i64 %call, ptr %retval, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %4 = load i64, ptr %retval, align 8
  ret i64 %4
}

; Function Attrs: noinline nounwind optnone
define internal i64 @internal_func(i32 noundef %n) #0 {
entry:
  %pgocount = load i64, ptr @__profc_file1.c_internal_func, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_file1.c_internal_func, align 8
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %1 = load i32, ptr %n.addr, align 4
  %conv = sext i32 %1 to i64
  ret i64 %conv
}

; Function Attrs: noinline nounwind optnone
define ptr @escape1() #0 {
entry:
  %pgocount = load i64, ptr @__profc_escape1, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_escape1, align 8
  ret ptr @internal_func
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #1

attributes #0 = { noinline nounwind optnone }
attributes #1 = { nounwind }

!0 = !{ptr @__profc_func1}
!1 = !{ptr @__profc_file1.c_internal_func}
!2 = !{ptr @__profc_escape1}

;--- f2.ll
target datalayout = "E-m:a-p:32:32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

@__llvm_profile_raw_version = weak hidden constant i64 72057594037927944
@__profc_func2 = private global [2 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_func2 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 -4377547752858689819, i64 146835647075900052, i32 sub (i32 ptrtoint (ptr @__profc_func2 to i32), i32 ptrtoint (ptr @__profd_func2 to i32)), ptr @func2.local, ptr null, i32 2, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !0
@__profc_file2.c_internal_func = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_file2.c_internal_func = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 4899437111831460578, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_file2.c_internal_func to i32), i32 ptrtoint (ptr @__profd_file2.c_internal_func to i32)), ptr @internal_func, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !1
@__profc_escape2 = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_escape2 = private global { i64, i64, i32, ptr, ptr, i32, [4 x i16] } { i64 3333263850724280696, i64 742261418966908927, i32 sub (i32 ptrtoint (ptr @__profc_escape2 to i32), i32 ptrtoint (ptr @__profd_escape2 to i32)), ptr @escape2.local, ptr null, i32 1, [4 x i16] zeroinitializer }, section "__llvm_prf_data", align 8, !pgo.associated !2
@__llvm_prf_nm = private constant [37 x i8] c"#\00func2\01file2.c:internal_func\01escape2", section "__llvm_prf_names", align 1
@llvm.used = appending global [4 x ptr] [ptr @__profd_func2, ptr @__profd_file2.c_internal_func, ptr @__profd_escape2, ptr @__llvm_prf_nm], section "llvm.metadata"
@__llvm_profile_filename = weak hidden constant [19 x i8] c"default_%m.profraw\00"

@func2.local = private alias i64 (i32), ptr @func2
@escape2.local = private alias ptr (), ptr @escape2

; Function Attrs: noinline nounwind optnone
define i64 @func2(i32 noundef %n) #0 {
entry:
  %retval = alloca i64, align 8
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp ult i32 %0, 30
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %pgocount = load i64, ptr @__profc_func2, align 8
  %1 = add i64 %pgocount, 1
  store i64 %1, ptr @__profc_func2, align 8
  store i64 0, ptr %retval, align 8
  br label %return

if.end:                                           ; preds = %entry
  %pgocount1 = load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc_func2, i32 0, i32 1), align 8
  %2 = add i64 %pgocount1, 1
  store i64 %2, ptr getelementptr inbounds ([2 x i64], ptr @__profc_func2, i32 0, i32 1), align 8
  %3 = load i32, ptr %n.addr, align 4
  %call = call i64 @internal_func(i32 noundef %3)
  store i64 %call, ptr %retval, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %4 = load i64, ptr %retval, align 8
  ret i64 %4
}

; Function Attrs: noinline nounwind optnone
define internal i64 @internal_func(i32 noundef %n) #0 {
entry:
  %pgocount = load i64, ptr @__profc_file2.c_internal_func, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_file2.c_internal_func, align 8
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %1 = load i32, ptr %n.addr, align 4
  %not = xor i32 %1, -1
  %add = add i32 %not, 1
  %conv = zext i32 %add to i64
  ret i64 %conv
}

; Function Attrs: noinline nounwind optnone
define ptr @escape2() #0 {
entry:
  %pgocount = load i64, ptr @__profc_escape2, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_escape2, align 8
  ret ptr @internal_func
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #1

attributes #0 = { noinline nounwind optnone }
attributes #1 = { nounwind }

!0 = !{ptr @__profc_func2}
!1 = !{ptr @__profc_file2.c_internal_func}
!2 = !{ptr @__profc_escape2}

; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_func1[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_func1[RW]
; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_file1.c_internal_func[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_file1.c_internal_func[RW]
; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_func2[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_func2[RW]
; CHECK-DAG:        .csect __llvm_prf_cnts.__profc_file2.c_internal_func[RW]
; CHECK-DAG:        .csect __llvm_prf_data.__profd_file2.c_internal_func[RW]
; CHECK-DAG:        .csect __llvm_prf_names[RO]


; CHECK:            .csect __llvm_prf_cnts.__profc_func1[RW],3
; CHECK-NEXT:       .ref __llvm_prf_names[RO]
; CHECK-NEXT:       .ref __llvm_prf_data.__profd_func1[RW]
; CHECK-NEXT:       .rename __llvm_prf_cnts.__profc_func1[RW],"__llvm_prf_cnts"

; CHECK:            .csect __llvm_prf_data.__profd_func1[RW],3
; CHECK-NEXT:       .rename __llvm_prf_data.__profd_func1[RW],"__llvm_prf_data"

; CHECK:            .csect __llvm_prf_cnts.__profc_file1.c_internal_func[RW]
; CHECK-NEXT:       .ref __llvm_prf_names[RO]
; CHECK-NEXT:       .ref __llvm_prf_data.__profd_file1.c_internal_func[RW]
; CHECK-NEXT:       .rename __llvm_prf_cnts.__profc_file1.c_internal_func[RW],"__llvm_prf_cnts"

; CHECK:            .csect __llvm_prf_data.__profd_file1.c_internal_func[RW]
; CHECK-NEXT:       .rename __llvm_prf_data.__profd_file1.c_internal_func[RW],"__llvm_prf_data"

; CHECK:            .csect __llvm_prf_cnts.__profc_func2[RW]
; CHECK-NEXT:       .ref __llvm_prf_names[RO]
; CHECK-NEXT:       .ref __llvm_prf_data.__profd_func2[RW]
; CHECK-NEXT:       .rename __llvm_prf_cnts.__profc_func2[RW],"__llvm_prf_cnts"

; CHECK:            .csect __llvm_prf_data.__profd_func2[RW]
; CHECK-NEXT:       .rename __llvm_prf_data.__profd_func2[RW],"__llvm_prf_data"

; CHECK:            .csect __llvm_prf_cnts.__profc_file2.c_internal_func[RW]
; CHECK-NEXT:       .ref __llvm_prf_names[RO]
; CHECK-NEXT:       .ref __llvm_prf_data.__profd_file2.c_internal_func[RW]
; CHECK-NEXT:       .rename __llvm_prf_cnts.__profc_file2.c_internal_func[RW],"__llvm_prf_cnts"

; CHECK:            .csect __llvm_prf_data.__profd_file2.c_internal_func[RW]
; CHECK-NEXT:       .rename __llvm_prf_data.__profd_file2.c_internal_func[RW],"__llvm_prf_data"
