;RUN: llc -march=sparc < %s | FileCheck %s

%struct.foo_t = type { i32, i32, i32 }

define weak void @make_foo(%struct.foo_t* noalias sret(%struct.foo_t) %agg.result, i32 %a, i32 %b, i32 %c) nounwind {
entry:
;CHECK-LABEL: make_foo:
;CHECK: ld [%sp+64], {{.+}}
;CHECK: jmp %o7+12
  %0 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %agg.result, i32 0, i32 0
  store i32 %a, i32* %0, align 4
  %1 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %agg.result, i32 0, i32 1
  store i32 %b, i32* %1, align 4
  %2 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %agg.result, i32 0, i32 2
  store i32 %c, i32* %2, align 4
  ret void
}

define i32 @test() nounwind {
entry:
;CHECK-LABEL: test:
;CHECK: call make_foo
;CHECK: st {{.+}}, [%sp+64]
;CHECK: unimp 12
  %f = alloca %struct.foo_t, align 8
  call void @make_foo(%struct.foo_t* noalias sret(%struct.foo_t) %f, i32 10, i32 20, i32 30) nounwind
  %0 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %f, i32 0, i32 0
  %1 = load i32, i32* %0, align 8
  %2 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %f, i32 0, i32 1
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds %struct.foo_t, %struct.foo_t* %f, i32 0, i32 2
  %5 = load i32, i32* %4, align 8
  %6 = add nsw i32 %3, %1
  %7 = add nsw i32 %6, %5
  ret i32 %7
}
