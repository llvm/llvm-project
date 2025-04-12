; RUN: opt < %s -passes='print<phi-values>' -disable-output 2>&1 | FileCheck %s

@X = common global i32 0

; CHECK-LABEL: PHI Values for function: simple
define void @simple(ptr %ptr, i1 %arg) {
entry:
  br i1 %arg, label %if, label %else

if:
  br label %end

else:
  br label %end

end:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
  %phi1 = phi i32 [ 0, %if ], [ 1, %else ]
; CHECK: PHI %phi2 has values:
; CHECK-DAG: @X
; CHECK-DAG: %ptr
  %phi2 = phi ptr [ @X, %if ], [ %ptr, %else ]
  ret void
}

; CHECK-LABEL: PHI Values for function: chain
define void @chain(i1 %arg) {
entry:
  br i1 %arg, label %if1, label %else1

if1:
  br label %middle

else1:
  br label %middle

middle:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
  %phi1 = phi i32 [ 0, %if1 ], [ 1, %else1 ]
  br i1 %arg, label %if2, label %else2

if2:
  br label %end

else2:
  br label %end

end:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
; CHECK-DAG: i32 2
  %phi2 = phi i32 [ %phi1, %if2 ], [ 2, %else2 ]
  ret void
}

; CHECK-LABEL: PHI Values for function: no_values
define void @no_values(i1 %arg) {
entry:
  ret void

unreachable:
; CHECK: PHI %phi has values:
; CHECK-DAG: NONE
  %phi = phi i32 [ %phi, %unreachable ]
  br label %unreachable
}

; CHECK-LABEL: PHI Values for function: simple_loop
define void @simple_loop(i1 %arg) {
entry:
  br label %loop

loop:
; CHECK: PHI %phi has values:
; CHECK-DAG: i32 0
  %phi = phi i32 [ 0, %entry ], [ %phi, %loop ]
  br i1 %arg, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: PHI Values for function: complex_loop
define void @complex_loop(i1 %arg) {
entry:
  br i1 %arg, label %loop, label %end

loop:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %then ]
  br i1 %arg, label %if, label %else

if:
  br label %then

else:
  br label %then

then:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
  %phi2 = phi i32 [ %phi1, %if ], [ 1, %else ]
  br i1 %arg, label %loop, label %end

end:
; CHECK: PHI %phi3 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
; CHECK-DAG: i32 2
  %phi3 = phi i32 [ 2, %entry ], [ %phi2, %then ]
  ret void
}

; CHECK-LABEL: PHI Values for function: strange_loop
define void @strange_loop(i1 %arg) {
entry:
  br i1 %arg, label %ifelse, label %inloop

loop:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
; CHECK-DAG: i32 2
; CHECK-DAG: i32 3
  %phi1 = phi i32 [ %phi3, %if ], [ 0, %else ], [ %phi2, %inloop ]
  br i1 %arg, label %inloop, label %end

inloop:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: i32 1
; CHECK-DAG: i32 2
; CHECK-DAG: i32 3
  %phi2 = phi i32 [ %phi1, %loop ], [ 1, %entry ]
  br i1 %arg, label %ifelse, label %loop

ifelse:
; CHECK: PHI %phi3 has values:
; CHECK-DAG: i32 2
; CHECK-DAG: i32 3
  %phi3 = phi i32 [ 2, %entry ], [ 3, %inloop ]
  br i1 %arg, label %if, label %else

if:
  br label %loop

else:
  br label %loop

end:
  ret void
}

; CHECK-LABEL: PHI Values for function: mutual_loops
define void @mutual_loops(i1 %arg) {
entry:
  br i1 %arg, label %loop1, label %loop2

loop1:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: 0
; CHECK-DAG: 1
; CHECK-DAG: 2
; CHECK-DAG: 3
; CHECK-DAG: 4
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %loop1.then ], [ %phi3, %loop2.if ]
  br i1 %arg, label %loop1.if, label %loop1.else

loop1.if:
  br i1 %arg, label %loop1.then, label %loop2

loop1.else:
  br label %loop1.then

loop1.then:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: 0
; CHECK-DAG: 1
; CHECK-DAG: 2
; CHECK-DAG: 3
; CHECK-DAG: 4
  %phi2 = phi i32 [ 1, %loop1.if ], [ %phi1, %loop1.else ]
  br i1 %arg, label %loop1, label %end

loop2:
; CHECK: PHI %phi3 has values:
; CHECK-DAG: 2
; CHECK-DAG: 3
; CHECK-DAG: 4
  %phi3 = phi i32 [ 2, %entry ], [ %phi4, %loop2.then ], [ 3, %loop1.if ]
  br i1 %arg, label %loop2.if, label %loop2.else

loop2.if:
  br i1 %arg, label %loop2.then, label %loop1

loop2.else:
  br label %loop2.then

loop2.then:
; CHECK: PHI %phi4 has values:
; CHECK-DAG: 2
; CHECK-DAG: 3
; CHECK-DAG: 4
  %phi4 = phi i32 [ 4, %loop2.if ], [ %phi3, %loop2.else ]
  br i1 %arg, label %loop2, label %end

end:
; CHECK: PHI %phi5 has values:
; CHECK-DAG: 0
; CHECK-DAG: 1
; CHECK-DAG: 2
; CHECK-DAG: 3
; CHECK-DAG: 4
  %phi5 = phi i32 [ %phi2, %loop1.then ], [ %phi4, %loop2.then ]
  ret void
}

; CHECK-LABEL: PHI Values for function: nested_loops_several_values
define void @nested_loops_several_values(i1 %arg) {
entry:
  br label %loop1

loop1:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: %add
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %loop2 ]
  br i1 %arg, label %loop2, label %end

loop2:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: %add
  %phi2 = phi i32 [ %phi1, %loop1 ], [ %phi3, %loop3 ]
  br i1 %arg, label %loop3, label %loop1

loop3:
; CHECK: PHI %phi3 has values:
; CHECK-DAG: i32 0
; CHECK-DAG: %add
  %phi3 = phi i32 [ %add, %loop3 ], [ %phi2, %loop2 ]
  %add = add i32 %phi3, 1
  br i1 %arg, label %loop3, label %loop2

end:
  ret void
}

; CHECK-LABEL: PHI Values for function: nested_loops_one_value
define void @nested_loops_one_value(i1 %arg) {
entry:
  br label %loop1

loop1:
; CHECK: PHI %phi1 has values:
; CHECK-DAG: i32 0
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %loop2 ]
  br i1 %arg, label %loop2, label %end

loop2:
; CHECK: PHI %phi2 has values:
; CHECK-DAG: i32 0
  %phi2 = phi i32 [ %phi1, %loop1 ], [ %phi3, %loop3 ]
  br i1 %arg, label %loop3, label %loop1

loop3:
; CHECK: PHI %phi3 has values:
; CHECK-DAG: i32 0
  %phi3 = phi i32 [ 0, %loop3 ], [ %phi2, %loop2 ]
  br i1 %arg, label %loop3, label %loop2

end:
  ret void
}
