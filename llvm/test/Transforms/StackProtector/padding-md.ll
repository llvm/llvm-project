; REQUIRES: aarch64-registered-target
; RUN: opt -mtriple=arm64-apple-macosx %s -passes='require<libcall-lowering-info>,stack-protector' -S | FileCheck %s

declare void @use(ptr)

%struct.padding = type { i32, [12 x i8] }
; CHECK-LABEL: define void @padding(
; CHECK-NOT:     call void @llvm.stackprotector
; CHECK:         ret void
define void @padding() ssp {
  %a = alloca %struct.padding, align 16, !stack-protector-padding !0
  call void @use(ptr %a)
  ret void
}
!0 = !{i64 16, i64 4, i64 12}

; Control: the same alloca without the metadata is protected.
%struct.no_md = type { i32, [12 x i8] }
; CHECK-LABEL: define void @no_md(
; CHECK:         call void @llvm.stackprotector
define void @no_md() ssp {
  %a = alloca %struct.no_md, align 16
  call void @use(ptr %a)
  ret void
}

; The range must cover the whole array.
%struct.partial = type { i32, [12 x i8] }
; CHECK-LABEL: define void @partial(
; CHECK:         call void @llvm.stackprotector
define void @partial() ssp {
  %a = alloca %struct.partial, align 16, !stack-protector-padding !1
  call void @use(ptr %a)
  ret void
}
!1 = !{i64 16, i64 8, i64 8}

; Ranges for an object of a different size are stale and ignored.
%struct.size_mismatch = type { i32, [12 x i8] }
; CHECK-LABEL: define void @size_mismatch(
; CHECK:         call void @llvm.stackprotector
define void @size_mismatch() ssp {
  %a = alloca %struct.size_mismatch, align 16, !stack-protector-padding !2
  call void @use(ptr %a)
  ret void
}
!2 = !{i64 32, i64 4, i64 12}

; Offsets apply to nested structs. Only the padding in the inner struct is covered by the metadata, the outer [16 x i8] is still a buffer.
%struct.nested_buffer.inner = type { i32, [12 x i8] }
%struct.nested_buffer = type { %struct.nested_buffer.inner, [16 x i8] }
; CHECK-LABEL: define void @nested_buffer(
; CHECK:         call void @llvm.stackprotector
define void @nested_buffer() ssp {
  %a = alloca %struct.nested_buffer, align 16, !stack-protector-padding !3
  call void @use(ptr %a)
  ret void
}
!3 = !{i64 32, i64 4, i64 12}

%struct.nested_padding.inner = type { i32, [12 x i8] }
%struct.nested_padding = type { %struct.nested_padding.inner, [16 x i8] }
; CHECK-LABEL: define void @nested_padding(
; CHECK-NOT:     call void @llvm.stackprotector
; CHECK:         ret void
define void @nested_padding() ssp {
  %a = alloca %struct.nested_padding, align 16, !stack-protector-padding !4
  call void @use(ptr %a)
  ret void
}
!4 = !{i64 32, i64 4, i64 12, i64 16, i64 16}

%struct.small_buffer_and_padding = type { i32, [4 x i8], [8 x i8] }
; CHECK-LABEL: define void @small_buffer_and_padding(
; CHECK-NOT:     call void @llvm.stackprotector
; CHECK:         ret void
define void @small_buffer_and_padding() ssp {
  %a = alloca %struct.small_buffer_and_padding, align 16, !stack-protector-padding !5
  call void @use(ptr %a)
  ret void
}
!5 = !{i64 16, i64 8, i64 8}

%struct.padding_and_buffer = type { i32, [12 x i8] }
; CHECK-LABEL: define void @padding_and_buffer(
; CHECK:         call void @llvm.stackprotector
define void @padding_and_buffer() ssp {
  %a = alloca %struct.padding_and_buffer, align 16, !stack-protector-padding !6
  %buf = alloca [16 x i8], align 1
  call void @use(ptr %a)
  call void @use(ptr %buf)
  ret void
}
!6 = !{i64 16, i64 4, i64 12}

%struct.struct_array = type { i32, [12 x i8] }
; CHECK-LABEL: define void @struct_array(
; CHECK:         call void @llvm.stackprotector
define void @struct_array() ssp {
  %a = alloca [4 x %struct.struct_array], align 16, !stack-protector-padding !7
  call void @use(ptr %a)
  ret void
}
!7 = !{i64 64, i64 4, i64 12}

%struct.padding_sspstrong = type { i32, [12 x i8] }
; CHECK-LABEL: define void @padding_sspstrong(
; CHECK:         call void @llvm.stackprotector
define void @padding_sspstrong() sspstrong {
  %a = alloca %struct.padding_sspstrong, align 16, !stack-protector-padding !8
  call void @use(ptr %a)
  ret void
}
!8 = !{i64 16, i64 4, i64 12}

%struct.padding_sspreq = type { i32, [12 x i8] }
; CHECK-LABEL: define void @padding_sspreq(
; CHECK:         call void @llvm.stackprotector
define void @padding_sspreq() sspreq {
  %a = alloca %struct.padding_sspreq, align 16, !stack-protector-padding !9
  call void @use(ptr %a)
  ret void
}
!9 = !{i64 16, i64 4, i64 12}

%struct.ignored_sspstrong = type { i32, [12 x i8] }
; CHECK-LABEL: define void @ignored_sspstrong(
; CHECK-NOT:     call void @llvm.stackprotector
; CHECK:         ret void
define void @ignored_sspstrong() sspstrong {
  %a = alloca %struct.ignored_sspstrong, align 16, !stack-protector !10, !stack-protector-padding !11
  call void @use(ptr %a)
  ret void
}
!10 = !{i32 0}
!11 = !{i64 16, i64 4, i64 12}
