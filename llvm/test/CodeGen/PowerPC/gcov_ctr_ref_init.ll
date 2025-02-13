; Tests if the __llvm_gcov_ctr section contains a .ref pseudo-op
; referring to the __llvm_covinit section.
; RUN: llc < %s | FileCheck --check-prefixes=CHECK,CHECK-RW %s
; RUN: llc -mxcoff-roptr < %s | FileCheck --check-prefixes=CHECK,CHECK-RO %s

target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"
target triple = "powerpc-ibm-aix"

; CHECK-RW: .csect __llvm_covinit[RW],3
; CHECK-RO: .csect __llvm_covinit[RO],3
; CHECK-NEXT:    .align  3                               # @__llvm_covinit_functions
; CHECK-NEXT: L..__llvm_covinit_functions:
; CHECK-NEXT:     .vbyte  4, __llvm_gcov_writeout[DS]
; CHECK-NEXT:     .vbyte  4, __llvm_gcov_reset[DS]
; CHECK:    .csect __llvm_gcov_ctr_section[RW],3
; CHECK-NEXT:    .lglobl __llvm_gcov_ctr                 # @_MergedGlobals
; CHECK-NEXT:    .lglobl __llvm_gcov_ctr.1
; CHECK-NEXT:    .align  3
; CHECK-NEXT: L.._MergedGlobals:
; CHECK-NEXT: __llvm_gcov_ctr:
; CHECK-NEXT:     .space  8
; CHECK-NEXT: __llvm_gcov_ctr.1:
; CHECK-NEXT:     .space  8
; CHECK:     .csect __llvm_gcov_ctr_section[RW],3
; CHECK-RW-NEXT:    .ref __llvm_covinit[RW]
; CHECK-RO-NEXT:    .ref __llvm_covinit[RO]

%emit_function_args_ty = type { i32, i32, i32 }
%emit_arcs_args_ty = type { i32, ptr }
%file_info = type { %start_file_args_ty, i32, ptr, ptr }
%start_file_args_ty = type { ptr, i32, i32 }

@__llvm_gcov_ctr = internal global [1 x i64] zeroinitializer, section "__llvm_gcov_ctr_section"
@__llvm_gcov_ctr.1 = internal global [1 x i64] zeroinitializer, section "__llvm_gcov_ctr_section"
@0 = private unnamed_addr constant [10 x i8] c"test.gcda\00", align 1
@__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant [2 x %emit_function_args_ty] [%emit_function_args_ty { i32 0, i32 1961870044, i32 -801444649 }, %emit_function_args_ty { i32 1, i32 1795396728, i32 -801444649 }]
@__llvm_internal_gcov_emit_arcs_args.0 = internal unnamed_addr constant [2 x %emit_arcs_args_ty] [%emit_arcs_args_ty { i32 1, ptr @__llvm_gcov_ctr }, %emit_arcs_args_ty { i32 1, ptr @__llvm_gcov_ctr.1 }]
@__llvm_internal_gcov_emit_file_info = internal unnamed_addr constant [1 x %file_info] [%file_info { %start_file_args_ty { ptr @0, i32 875575338, i32 -801444649 }, i32 2, ptr @__llvm_internal_gcov_emit_function_args.0, ptr @__llvm_internal_gcov_emit_arcs_args.0 }]
@__llvm_covinit_functions = private constant { ptr, ptr } { ptr @__llvm_gcov_writeout, ptr @__llvm_gcov_reset }, section "__llvm_covinit", align 8

define i32 @bar() {
entry:
  %gcov_ctr = load i64, ptr @__llvm_gcov_ctr, align 8
  %0 = add i64 %gcov_ctr, 1
  store i64 %0, ptr @__llvm_gcov_ctr, align 8
  ret i32 1
}

define i32 @main() {
entry:
  %gcov_ctr = load i64, ptr @__llvm_gcov_ctr.1, align 8
  %0 = add i64 %gcov_ctr, 1
  store i64 %0, ptr @__llvm_gcov_ctr.1, align 8
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 @bar()
  %sub = sub nsw i32 %call, 1
  ret i32 %sub
}

define internal void @__llvm_gcov_writeout() unnamed_addr {
entry:
  br label %file.loop.header

file.loop.header:                                 ; preds = %file.loop.latch, %entry
  %file_idx = phi i32 [ 0, %entry ], [ %next_file_idx, %file.loop.latch ]
  %0 = getelementptr inbounds [1 x %file_info], ptr @__llvm_internal_gcov_emit_file_info, i32 0, i32 %file_idx
  %start_file_args = getelementptr inbounds nuw %file_info, ptr %0, i32 0, i32 0
  %1 = getelementptr inbounds nuw %start_file_args_ty, ptr %start_file_args, i32 0, i32 0
  %filename = load ptr, ptr %1, align 4
  %2 = getelementptr inbounds nuw %start_file_args_ty, ptr %start_file_args, i32 0, i32 1
  %version = load i32, ptr %2, align 4
  %3 = getelementptr inbounds nuw %start_file_args_ty, ptr %start_file_args, i32 0, i32 2
  %stamp = load i32, ptr %3, align 4
  call void @llvm_gcda_start_file(ptr %filename, i32 %version, i32 %stamp)
  %4 = getelementptr inbounds nuw %file_info, ptr %0, i32 0, i32 1
  %num_ctrs = load i32, ptr %4, align 4
  %5 = getelementptr inbounds nuw %file_info, ptr %0, i32 0, i32 2
  %emit_function_args = load ptr, ptr %5, align 4
  %6 = getelementptr inbounds nuw %file_info, ptr %0, i32 0, i32 3
  %emit_arcs_args = load ptr, ptr %6, align 4
  %7 = icmp slt i32 0, %num_ctrs
  br i1 %7, label %counter.loop.header, label %file.loop.latch

counter.loop.header:                              ; preds = %counter.loop.header, %file.loop.header
  %ctr_idx = phi i32 [ 0, %file.loop.header ], [ %15, %counter.loop.header ]
  %8 = getelementptr inbounds %emit_function_args_ty, ptr %emit_function_args, i32 %ctr_idx
  %9 = getelementptr inbounds nuw %emit_function_args_ty, ptr %8, i32 0, i32 0
  %ident = load i32, ptr %9, align 4
  %10 = getelementptr inbounds nuw %emit_function_args_ty, ptr %8, i32 0, i32 1
  %func_checkssum = load i32, ptr %10, align 4
  %11 = getelementptr inbounds nuw %emit_function_args_ty, ptr %8, i32 0, i32 2
  %cfg_checksum = load i32, ptr %11, align 4
  call void @llvm_gcda_emit_function(i32 %ident, i32 %func_checkssum, i32 %cfg_checksum)
  %12 = getelementptr inbounds %emit_arcs_args_ty, ptr %emit_arcs_args, i32 %ctr_idx
  %13 = getelementptr inbounds nuw %emit_arcs_args_ty, ptr %12, i32 0, i32 0
  %num_counters = load i32, ptr %13, align 4
  %14 = getelementptr inbounds nuw %emit_arcs_args_ty, ptr %12, i32 0, i32 1
  %counters = load ptr, ptr %14, align 4
  call void @llvm_gcda_emit_arcs(i32 %num_counters, ptr %counters)
  %15 = add i32 %ctr_idx, 1
  %16 = icmp slt i32 %15, %num_ctrs
  br i1 %16, label %counter.loop.header, label %file.loop.latch

file.loop.latch:                                  ; preds = %counter.loop.header, %file.loop.header
  call void @llvm_gcda_summary_info()
  call void @llvm_gcda_end_file()
  %next_file_idx = add i32 %file_idx, 1
  %17 = icmp slt i32 %next_file_idx, 1
  br i1 %17, label %file.loop.header, label %exit

exit:                                             ; preds = %file.loop.latch
  ret void
}

declare void @llvm_gcda_start_file(ptr, i32, i32)

declare void @llvm_gcda_emit_function(i32, i32, i32)

declare void @llvm_gcda_emit_arcs(i32, ptr)

declare void @llvm_gcda_summary_info()

declare void @llvm_gcda_end_file()

define internal void @__llvm_gcov_reset() unnamed_addr {
entry:
  call void @llvm.memset.p0.i64(ptr @__llvm_gcov_ctr, i8 0, i64 8, i1 false)
  call void @llvm.memset.p0.i64(ptr @__llvm_gcov_ctr.1, i8 0, i64 8, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)


