; KMSAN instrumentation tests
; RUN: opt < %s -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s             \
; RUN: -check-prefixes=CHECK
; RUN: opt < %s -msan -msan-kernel=1 -S | FileCheck %s -check-prefixes=CHECK

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check the instrumentation prologue.
define void @Empty() nounwind uwtable sanitize_memory {
entry:
  ret void
}

; CHECK-LABEL: @Empty
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; %param_shadow:
; CHECK: getelementptr {{.*}} i32 0, i32 0
; %retval_shadow:
; CHECK: getelementptr {{.*}} i32 0, i32 1
; %va_arg_shadow:
; CHECK: getelementptr {{.*}} i32 0, i32 2
; %va_arg_origin:
; CHECK: getelementptr {{.*}} i32 0, i32 3
; %va_arg_overflow_size:
; CHECK: getelementptr {{.*}} i32 0, i32 4
; %param_origin:
; CHECK: getelementptr {{.*}} i32 0, i32 5
; %retval_origin:
; CHECK: getelementptr {{.*}} i32 0, i32 6
; CHECK: entry.split:

; Check instrumentation of stores

define void @Store1(i8* nocapture %p, i8 %x) nounwind uwtable sanitize_memory {
entry:
  store i8 %x, i8* %p
  ret void
}

; CHECK-LABEL: @Store1
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: [[BASE2:%[0-9]+]] = ptrtoint {{.*}} [[PARAM_SHADOW]]
; CHECK: [[BASE:%[0-9]+]] = ptrtoint {{.*}} [[PARAM_SHADOW]]
; CHECK: [[SHADOW:%[a-z0-9_]+]] = inttoptr {{.*}} [[BASE]]
; Load the shadow of %p and check it
; CHECK: load i64, i64* [[SHADOW]]
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_metadata_ptr_for_store_1(i8* %p)
; CHECK: store i8
; If the new shadow is non-zero, jump to __msan_chain_origin()
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_chain_origin
; Storing origin here:
; CHECK: store i32
; CHECK: br label
; CHECK: {{^[0-9]+}}:
; CHECK: store i8
; CHECK: ret void

define void @Store2(i16* nocapture %p, i16 %x) nounwind uwtable sanitize_memory {
entry:
  store i16 %x, i16* %p
  ret void
}

; CHECK-LABEL: @Store2
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: [[REG:%[0-9]+]] = bitcast i16* %p to i8*
; CHECK: @__msan_metadata_ptr_for_store_2(i8* [[REG]])
; CHECK: store i16
; If the new shadow is non-zero, jump to __msan_chain_origin()
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_chain_origin
; Storing origin here:
; CHECK: store i32
; CHECK: br label
; CHECK: {{^[0-9]+}}:
; CHECK: store i16
; CHECK: ret void


define void @Store4(i32* nocapture %p, i32 %x) nounwind uwtable sanitize_memory {
entry:
  store i32 %x, i32* %p
  ret void
}

; CHECK-LABEL: @Store4
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i32
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: [[REG:%[0-9]+]] = bitcast i32* %p to i8*
; CHECK: @__msan_metadata_ptr_for_store_4(i8* [[REG]])
; CHECK: store i32
; If the new shadow is non-zero, jump to __msan_chain_origin()
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_chain_origin
; Storing origin here:
; CHECK: store i32
; CHECK: br label
; CHECK: {{^[0-9]+}}:
; CHECK: store i32
; CHECK: ret void

define void @Store8(i64* nocapture %p, i64 %x) nounwind uwtable sanitize_memory {
entry:
  store i64 %x, i64* %p
  ret void
}

; CHECK-LABEL: @Store8
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: [[REG:%[0-9]+]] = bitcast i64* %p to i8*
; CHECK: @__msan_metadata_ptr_for_store_8(i8* [[REG]])
; CHECK: store i64
; If the new shadow is non-zero, jump to __msan_chain_origin()
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_chain_origin
; Storing origin here:
; CHECK: store i64
; CHECK: br label
; CHECK: {{^[0-9]+}}:
; CHECK: store i64
; CHECK: ret void

define void @Store16(i128* nocapture %p, i128 %x) nounwind uwtable sanitize_memory {
entry:
  store i128 %x, i128* %p
  ret void
}

; CHECK-LABEL: @Store16
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: [[REG:%[0-9]+]] = bitcast i128* %p to i8*
; CHECK: @__msan_metadata_ptr_for_store_n(i8* [[REG]], i64 16)
; CHECK: store i128
; If the new shadow is non-zero, jump to __msan_chain_origin()
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; CHECK: @__msan_chain_origin
; Storing origin here:
; CHECK: store i64
; CHECK: br label
; CHECK: {{^[0-9]+}}:
; CHECK: store i128
; CHECK: ret void


; Check instrumentation of loads

define i8 @Load1(i8* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  %0 = load i8, i8* %p
  ret i8 %0
}

; CHECK-LABEL: @Load1
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; Load the value from %p. This is done before accessing the shadow
; to ease atomic handling.
; CHECK: load i8
; CHECK: @__msan_metadata_ptr_for_load_1(i8* %p)
; Load the shadow and origin.
; CHECK: load i8
; CHECK: load i32


define i16 @Load2(i16* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  %0 = load i16, i16* %p
  ret i16 %0
}

; CHECK-LABEL: @Load2
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; Load the value from %p. This is done before accessing the shadow
; to ease atomic handling.
; CHECK: load i16
; CHECK: [[REG:%[0-9]+]] = bitcast i16* %p to i8*
; CHECK: @__msan_metadata_ptr_for_load_2(i8* [[REG]])
; Load the shadow and origin.
; CHECK: load i16
; CHECK: load i32


define i32 @Load4(i32* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  %0 = load i32, i32* %p
  ret i32 %0
}

; CHECK-LABEL: @Load4
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; Load the value from %p. This is done before accessing the shadow
; to ease atomic handling.
; CHECK: load i32
; CHECK: [[REG:%[0-9]+]] = bitcast i32* %p to i8*
; CHECK: @__msan_metadata_ptr_for_load_4(i8* [[REG]])
; Load the shadow and origin.
; CHECK: load i32
; CHECK: load i32

define i64 @Load8(i64* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  %0 = load i64, i64* %p
  ret i64 %0
}

; CHECK-LABEL: @Load8
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; Load the value from %p. This is done before accessing the shadow
; to ease atomic handling.
; CHECK: load i64
; CHECK: [[REG:%[0-9]+]] = bitcast i64* %p to i8*
; CHECK: @__msan_metadata_ptr_for_load_8(i8* [[REG]])
; Load the shadow and origin.
; CHECK: load i64
; CHECK: load i32

define i128 @Load16(i128* nocapture %p) nounwind uwtable sanitize_memory {
entry:
  %0 = load i128, i128* %p
  ret i128 %0
}

; CHECK-LABEL: @Load16
; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: entry.split:
; CHECK: ptrtoint {{.*}} [[PARAM_SHADOW]]
; Load the shadow of %p and check it
; CHECK: load i64
; CHECK: icmp
; CHECK: br i1
; CHECK: {{^[0-9]+}}:
; Load the value from %p. This is done before accessing the shadow
; to ease atomic handling.
; CHECK: load i128
; CHECK: [[REG:%[0-9]+]] = bitcast i128* %p to i8*
; CHECK: @__msan_metadata_ptr_for_load_n(i8* [[REG]], i64 16)
; Load the shadow and origin.
; CHECK: load i128
; CHECK: load i32


; Test kernel-specific va_list instrumentation

%struct.__va_list_tag = type { i32, i32, i8*, i8* }
declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_end(i8*)
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
declare dso_local i32 @VAListFn(i8*, %struct.__va_list_tag*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define dso_local i32 @VarArgFn(i8* %fmt, ...) local_unnamed_addr sanitize_memory #0 {
entry:
  %args = alloca [1 x %struct.__va_list_tag], align 16
  %0 = bitcast [1 x %struct.__va_list_tag]* %args to i8*
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %args, i64 0, i64 0
  call void @llvm.va_start(i8* nonnull %0)
  %call = call i32 @VAListFn(i8* %fmt, %struct.__va_list_tag* nonnull %arraydecay)
  call void @llvm.va_end(i8* nonnull %0)
  ret i32 %call
}

; Kernel is built without SSE support.
attributes #0 = { "target-features"="+fxsr,+x87,-sse" }

; CHECK-LABEL: @VarArgFn
; CHECK: @__msan_get_context_state()
; CHECK: [[VA_ARG_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 2
; CHECK: [[VA_ARG_ORIGIN:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 3
; CHECK: [[VA_ARG_OVERFLOW_SIZE:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 4

; CHECK: entry.split:
; CHECK: [[OSIZE:%[0-9]+]] = load i64, i64* [[VA_ARG_OVERFLOW_SIZE]]
; Register save area is 48 bytes for non-SSE builds.
; CHECK: [[SIZE:%[0-9]+]] = add i64 48, [[OSIZE]]
; CHECK: [[SHADOWS:%[0-9]+]] = alloca i8, i64 [[SIZE]]
; CHECK: [[VA_ARG_SHADOW]]
; CHECK: call void @llvm.memcpy{{.*}}(i8* align 8 [[SHADOWS]], {{.*}}, i64 [[SIZE]]
; CHECK: [[ORIGINS:%[0-9]+]] = alloca i8, i64 [[SIZE]]
; CHECK: [[VA_ARG_ORIGIN]]
; CHECK: call void @llvm.memcpy{{.*}}(i8* align 8 [[ORIGINS]], {{.*}}, i64 [[SIZE]]
; CHECK: call i32 @VAListFn

; Function Attrs: nounwind uwtable
define dso_local void @VarArgCaller() local_unnamed_addr sanitize_memory {
entry:
  %call = tail call i32 (i8*, ...) @VarArgFn(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 123)
  ret void
}

; CHECK-LABEL: @VarArgCaller

; CHECK: entry:
; CHECK: @__msan_get_context_state()
; CHECK: [[PARAM_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 0
; CHECK: [[VA_ARG_SHADOW:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 2
; CHECK: [[VA_ARG_OVERFLOW_SIZE:%[a-z0-9_]+]] = getelementptr {{.*}} i32 0, i32 4

; CHECK: entry.split:
; CHECK: [[PARAM_SI:%[_a-z0-9]+]] = ptrtoint {{.*}} [[PARAM_SHADOW]]
; CHECK: [[ARG1_S:%[_a-z0-9]+]] = inttoptr i64 [[PARAM_SI]] to i64*
; First argument is initialized
; CHECK: store i64 0, i64* [[ARG1_S]]

; Dangling cast of va_arg_shadow[0], unused because the first argument is fixed.
; CHECK: [[VA_CAST0:%[_a-z0-9]+]] = ptrtoint {{.*}} [[VA_ARG_SHADOW]] to i64

; CHECK: [[VA_CAST1:%[_a-z0-9]+]] = ptrtoint {{.*}} [[VA_ARG_SHADOW]] to i64
; CHECK: [[ARG1_SI:%[_a-z0-9]+]] = add i64 [[VA_CAST1]], 8
; CHECK: [[PARG1_S:%[_a-z0-9]+]] = inttoptr i64 [[ARG1_SI]] to i32*

; Shadow for 123 is 0.
; CHECK: store i32 0, i32* [[ARG1_S]]

; CHECK: store i64 0, i64* [[VA_ARG_OVERFLOW_SIZE]]
; CHECK: call i32 (i8*, ...) @VarArgFn({{.*}} @.str{{.*}} i32 123)
