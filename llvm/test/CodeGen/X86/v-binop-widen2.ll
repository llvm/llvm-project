; RUN: llc -march=x86 -mcpu=generic -mattr=+sse < %s | FileCheck %s
; RUN: llc -march=x86 -mcpu=atom -mattr=+sse < %s | FileCheck -check-prefix=ATOM %s

%vec = type <6 x float>
; CHECK: divss
; CHECK: divss
; CHECK: divps

; Scheduler causes a different instruction order to be produced on Intel Atom
; ATOM: divps
; ATOM: divss
; ATOM: divss

define %vec @vecdiv( %vec %p1, %vec %p2)
{
  %result = fdiv %vec %p1, %p2
  ret %vec %result
}

@a = constant %vec < float 2.0, float 4.0, float 8.0, float 16.0, float 32.0, float 64.0 >
@b = constant %vec < float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0 >

; Expected result: < 1.0, 2.0, 4.0, ..., 2.0^(n-1) >
; main() returns 0 if the result is expected and 1 otherwise
; to execute, use llvm-as < %s | lli
define i32 @main() nounwind {
entry:
  %avec = load %vec* @a
  %bvec = load %vec* @b

  %res = call %vec @vecdiv(%vec %avec, %vec %bvec)
  br label %loop
loop:
  %idx = phi i32 [0, %entry], [%nextInd, %looptail]
  %expected = phi float [1.0, %entry], [%nextExpected, %looptail]
  %elem = extractelement %vec %res, i32 %idx
  %expcmp = fcmp oeq float %elem, %expected
  br i1 %expcmp, label %looptail, label %return
looptail:
  %nextExpected = fmul float %expected, 2.0
  %nextInd = add i32 %idx, 1
  %cmp = icmp slt i32 %nextInd, 6
  br i1 %cmp, label %loop, label %return
return:
  %retval = phi i32 [0, %looptail], [1, %loop]
  ret i32 %retval
}
