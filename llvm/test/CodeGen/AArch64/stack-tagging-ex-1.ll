; RUN: opt -S -aarch64-stack-tagging %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define  void @f() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
start:
; CHECK-LABEL: start:
  %a = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %a) #2
; CHECK: call void @llvm.aarch64.settag(ptr %a.tag, i64 48)
  %b = alloca i8, i32 48, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %b) #2
; CHECK: call void @llvm.aarch64.settag(ptr %b.tag, i64 48)
  invoke void @g (ptr nonnull %a, ptr nonnull %b) to label %next0 unwind label %lpad0
; CHECK-NOT: settag

next0:
; CHECK-LABEL: next0:
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %a)
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %b)
  br label %exit
; CHECK-NOT: settag

lpad0:
; CHECK-LABEL: lpad0:
  %pad0v = landingpad { ptr, i32 } catch ptr null
  %v = extractvalue { ptr, i32 } %pad0v, 0
  %x = call ptr @__cxa_begin_catch(ptr %v) #2
  invoke void @__cxa_end_catch() to label %next1 unwind label %lpad1
; CHECK-NOT: settag

next1:
; CHECK-LABEL: next1:
  br label %exit
; CHECK-NOT: settag

lpad1:
; CHECK-LABEL: lpad1:
; CHECK-DAG: call void @llvm.aarch64.settag(ptr %a, i64 48)
; CHECK-DAG: call void @llvm.aarch64.settag(ptr %b, i64 48)
  %pad1v = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %pad1v

exit:
; CHECK-LABEL: exit:
; CHECK-DAG: call void @llvm.aarch64.settag(ptr %a, i64 48)
; CHECK-DAG: call void @llvm.aarch64.settag(ptr %b, i64 48)
  ret void
; CHECK: ret void
}

declare void @g(ptr, ptr) #0

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare dso_local void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { sanitize_memtag "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="preserve-sign" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+mte,+neon,+v8.5a" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }
