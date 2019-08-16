; RUN: opt -S -passes=spec-phis %s

; This testcase crashes during the speculate around PHIs pass. The pass however
; results in no changes.

define i32 @test1() {
entry:
  callbr void asm sideeffect "", "X,X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@test1, %return), i8* blockaddress(@test1, %f))
          to label %asm.fallthrough [label %return, label %f]

asm.fallthrough:
  br label %return

f:
  br label %return

return:
  %retval.0 = phi i32 [ 0, %f ], [ 1, %asm.fallthrough ], [ 1, %entry ]
  ret i32 %retval.0
}

define void @test2() {
entry:
  br label %tailrecurse

tailrecurse:
  %call = tail call i32 @test3()
  %tobool1 = icmp eq i32 %call, 0
  callbr void asm sideeffect "", "X,X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@test2, %test1.exit), i8* blockaddress(@test2, %f.i))
          to label %if.end6 [label %test1.exit, label %f.i]

f.i:
  br label %test1.exit

test1.exit:
  %retval.0.i = phi i1 [ false, %f.i ], [ true, %tailrecurse ]
  %brmerge = or i1 %tobool1, %retval.0.i
  br i1 %brmerge, label %if.end6, label %tailrecurse

if.end6:
  ret void
}

declare i32 @test3()
