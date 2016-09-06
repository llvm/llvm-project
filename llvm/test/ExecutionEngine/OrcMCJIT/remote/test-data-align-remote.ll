; RUN:  %lli -jit-kind=orc-mcjit -remote-mcjit -O0 -mcjit-remote-process=lli-child-target%exeext %s
; XFAIL: mingw32,win32
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

; Check that a variable is always aligned as specified.

@var = global i32 0, align 32
define i32 @main() nounwind {
  %addr = ptrtoint i32* @var to i64
  %mask = and i64 %addr, 31
  %tst = icmp eq i64 %mask, 0
  br i1 %tst, label %good, label %bad
good:
  ret i32 0
bad:
  ret i32 1
}
