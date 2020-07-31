; RUN: opt < %s -instcombine -attributor-cgscc -print-callgraph -disable-output -verify 2>&1 | FileCheck %s

; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK:   CS<None> calls function 'dead_fork_call'
; CHECK:   CS<None> calls function 'd'
; CHECK:   CS<None> calls function '__kmpc_fork_call'
; CHECK:   CS<None> calls function 'live_fork_call'
; CHECK:   CS<None> calls function '.omp_outlined..1'
;
; CHECK: Call graph node for function: '.omp_outlined..1'<<{{.*}}>>  #uses=3
; CHECK:   CS<{{.*}}> calls function 'd'
;
; CHECK: Call graph node for function: '__kmpc_fork_call'<<{{.*}}>>  #uses=3
; CHECK:   CS<None> calls external node
;
; CHECK: Call graph node for function: 'd'<<{{.*}}>>  #uses=2
; CHECK:   CS<None> calls external node
;
; CHECK: Call graph node for function: 'dead_fork_call'<<{{.*}}>>  #uses=1
;
; CHECK: Call graph node for function: 'dead_fork_call2'<<{{.*}}>>  #uses=0
; CHECK:   CS<{{.*}}> calls function '__kmpc_fork_call'
; CHECK:   CS<None> calls function '.omp_outlined..1'
;
; CHECK: Call graph node for function: 'live_fork_call'<<{{.*}}>>  #uses=1
; CHECK:   CS<{{.*}}> calls function '__kmpc_fork_call'
; CHECK:   CS<None> calls function '.omp_outlined..1'


%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8

define dso_local void @dead_fork_call() {
entry:
  br i1 true, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  call void @dead_fork_call2()
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..0 to void (i32*, i32*, ...)*))
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define internal void @dead_fork_call2() {
entry:
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..1 to void (i32*, i32*, ...)*))
  ret void
}

define internal void @.omp_outlined..0(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  ret void
}

declare !callback !2 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

define dso_local void @live_fork_call() {
entry:
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..1 to void (i32*, i32*, ...)*))
  ret void
}

define internal void @.omp_outlined..1(i32* noalias %.global_tid., i32* noalias %.bound_tid.) {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void (...) @d()
  ret void
}

declare dso_local void @d(...)

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0"}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
