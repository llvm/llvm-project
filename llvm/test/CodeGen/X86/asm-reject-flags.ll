; RUN: not llc -filetype=null -mtriple=x86_64-unknown-unknown %s 2>&1 | FileCheck %s

; GH225033: {flags} is only valid as a clobber.

; CHECK: error: could not allocate output register for constraint '{flags}'
define x86_fp80 @flags_f80_output(x86_fp80 %x) {
  %r = call x86_fp80 asm "fabs", "={flags},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: could not allocate output register for constraint '{flags}'
define i32 @flags_i32_output(i32 %x) {
  %r = call i32 asm "mov $1, $0", "={flags},r,~{dirflag},~{fpsr},~{flags}"(i32 %x)
  ret i32 %r
}

; CHECK: error: could not allocate input reg for constraint '{flags}'
define void @flags_i32_input(i32 %x) {
  call void asm sideeffect "", "{flags},~{dirflag},~{fpsr},~{flags}"(i32 %x)
  ret void
}
