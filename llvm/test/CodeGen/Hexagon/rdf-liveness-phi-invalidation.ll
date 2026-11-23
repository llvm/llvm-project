; RUN: llc -march=hexagon -mcpu=hexagonv68 -mattr=+hvx-length128b -O3 < %s -o /dev/null
;
; Check that RDF liveness computation does not crash due to DenseMap
; iterator/reference invalidation. The function computePhiInfo holds a
; reference to a DenseMap value (RealUseMap[PA.Id]) while the inner loop
; may insert new entries into the same DenseMap, potentially triggering
; a rehash that invalidates the reference.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-linux-musl"

%struct.lua_TValue = type { %union.Value, i32 }
%union.Value = type { double }

define void @luaV_execute(ptr noundef %L) #0 {
entry:
  br label %for.cond

for.body1248.epil:
  %j1210.01943.epil = phi i32 [ 0, %sw.bb1205 ], [ 1, %for.body1248.epil ]
  %epil.iter = phi i32 [ 0, %sw.bb1205 ], [ %epil.iter.next, %for.body1248.epil ]
  %0 = load ptr, ptr %9, align 4
  %add.ptr1255.epil = getelementptr inbounds nuw %struct.lua_TValue, ptr %0, i32 %j1210.01943.epil
  %add.ptr1257.epil = getelementptr inbounds nuw %struct.lua_TValue, ptr null, i32 %j1210.01943.epil
  %1 = load i64, ptr %add.ptr1255.epil, align 8
  store i64 %1, ptr %add.ptr1257.epil, align 8
  %2 = getelementptr inbounds nuw %struct.lua_TValue, ptr null, i32 %j1210.01943.epil
  %tt1261.epil = getelementptr inbounds nuw i8, ptr %2, i32 8
  store i32 0, ptr %tt1261.epil, align 8
  %inc1267.epil = add nuw nsw i32 0, 1
  %epil.iter.next = add i32 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i32 %epil.iter.next, 7
  br i1 %epil.iter.cmp.not, label %for.cond, label %for.body1248.epil

for.cond:
  %base.0 = phi ptr [ null, %entry ], [ %base.1, %sw.bb560 ], [ null, %if.end14 ], [ null, %sw.bb257 ], [ null, %sw.bb973 ], [ null, %for.body1248.epil ], [ null, %sw.bb309 ], [ null, %sw.bb209 ]
  %3 = load i32, ptr null, align 4
  br label %if.end14

if.end14:
  %base.1 = phi ptr [ %base.0, %for.cond ]
  %and18 = and i32 %3, 63
  switch i32 %and18, label %for.cond [
    i32 37, label %sw.bb1205
    i32 32, label %sw.bb973
    i32 20, label %sw.bb560
    i32 14, label %sw.bb309
    i32 13, label %sw.bb257
    i32 12, label %sw.bb209
  ]

sw.bb209:
  %4 = load double, ptr null, align 8
  %add = fadd double %4, 0.000000e+00
  store double %add, ptr %base.1, align 8
  br label %for.cond

sw.bb257:
  %5 = load double, ptr null, align 8
  %sub = fsub double 0.000000e+00, %5
  store double %sub, ptr %base.1, align 8
  br label %for.cond

sw.bb309:
  %6 = load double, ptr null, align 8
  %mul = fmul double 0.000000e+00, %6
  store double %mul, ptr %base.1, align 8
  br label %for.cond

sw.bb560:
  %7 = load i32, ptr null, align 4
  %conv576 = uitofp i32 %7 to double
  store double %conv576, ptr %base.1, align 8
  br label %for.cond

sw.bb973:
  %8 = load double, ptr null, align 8
  %sub1011 = fsub double 0.000000e+00, %8
  store double %sub1011, ptr %base.1, align 8
  br label %for.cond

sw.bb1205:
  %9 = load ptr, ptr null, align 4
  br label %for.body1248.epil
}

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv68" "target-features"="+v68,-long-calls" }
