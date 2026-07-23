; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

declare float @func()

%struct = type { i32, float }

define void @test(ptr %ptr) {
  ; CHECK: nofpclass is only for loads
  call float @func(), !nofpclass !{i32 3}

  ; CHECK: nofpclass only applies to floating-point typed loads
  load i32, ptr %ptr, align 4, !nofpclass !{i32 3}

  ; CHECK: nofpclass only applies to floating-point typed loads
  load <2 x i32>, ptr %ptr, align 8, !nofpclass !{i32 3}

  ; CHECK: nofpclass only applies to floating-point typed loads
  load %struct, ptr %ptr, align 4, !nofpclass !{i32 3}

  ; CHECK: nofpclass must have exactly one entry
  %load = load float, ptr %ptr, align 4, !nofpclass !{}

  ; CHECK: nofpclass must have exactly one entry
  load float, ptr %ptr, align 4, !nofpclass !{i32 1, i32 2}

  ; CHECK: nofpclass entry must be a constant i32
  load float, ptr %ptr, align 4, !nofpclass !{i64 1}

  ; CHECK: nofpclass entry must be a constant i32
  load float, ptr %ptr, align 4, !nofpclass !{float 1.0}

  ; CHECK: nofpclass entry must be a constant i32
  load float, ptr %ptr, align 4, !nofpclass !{!"foo"}

  ; CHECK: nofpclass entry must be a constant i32
  load float, ptr %ptr, align 4, !nofpclass !{ptr @test}

  ; CHECK: 'nofpclass' must have at least one test bit set
  load float, ptr %ptr, align 4, !nofpclass !{i32 0}

  ; CHECK: Invalid value for 'nofpclass' test mask
  load float, ptr %ptr, align 4, !nofpclass !{i32 1024}

  ret void
}
