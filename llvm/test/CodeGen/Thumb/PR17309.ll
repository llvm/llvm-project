; RUN: llc -mtriple thumbv5-none-linux-gnueabi < %s | FileCheck %s

%struct.C = type { [1000 x i8] }
%struct.S = type { [1000 x i16] }
%struct.I = type { [1000 x i32] }

;CHECK-LABEL: pass_C:
;CHECK-NOT: ldrb    r{{[0-9]+}}, [{{.*}}], #1
;CHECK-NOT: strb    r{{[0-9]+}}, [{{.*}}], #1
define void @pass_C() #0 {
entry:
  %c = alloca %struct.C, align 1
  %0 = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1000, i8* %0) #1
  call void @use_C(%struct.C* byval %c) #3
  call void @llvm.lifetime.end.p0i8(i64 1000, i8* %0) #1
  ret void
}

;CHECK-LABEL: pass_S:
;CHECK-NOT: ldrh    r{{[0-9]+}}, [{{.*}}], #2
;CHECK-NOT: strh    r{{[0-9]+}}, [{{.*}}], #2
define void @pass_S() #0 {
entry:
  %s = alloca %struct.S, align 2
  %0 = bitcast %struct.S* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2000, i8* %0) #1
  call void @use_S(%struct.S* byval %s) #3
  call void @llvm.lifetime.end.p0i8(i64 2000, i8* %0) #1
  ret void
}

;CHECK-LABEL: pass_I:
;CHECK-NOT: ldr     r{{[0-9]+}}, [{{.*}}], #4
;CHECK-NOT: str     r{{[0-9]+}}, [{{.*}}], #4
define void @pass_I() #0 {
entry:
  %i = alloca %struct.I, align 4
  %0 = bitcast %struct.I* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %0) #1
  call void @use_I(%struct.I* byval %i) #3
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %0) #1
  ret void
}

declare void @use_C(%struct.C* byval) #2
declare void @use_S(%struct.S* byval) #2
declare void @use_I(%struct.I* byval) #2

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1


attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optsize }
