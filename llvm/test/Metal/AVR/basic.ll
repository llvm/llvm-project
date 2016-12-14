; RUN: avrlit %s

; CHECK-LABEL: test
define i16 @test() {
  ; CHECK-NEXT: return 1357
  ret i16 1357
}
