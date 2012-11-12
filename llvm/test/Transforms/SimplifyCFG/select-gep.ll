; RUN: opt -S -simplifycfg %s | FileCheck %s

define i8* @test1(i8* %x, i64 %y) nounwind {
entry:
  %tmp1 = load i8* %x, align 1
  %cmp = icmp eq i8 %tmp1, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %incdec.ptr = getelementptr inbounds i8* %x, i64 %y
  br label %if.end

if.end:
  %x.addr = phi i8* [ %incdec.ptr, %if.then ], [ %x, %entry ]
  ret i8* %x.addr

; CHECK: @test1
; CHECK-NOT: select
; CHECK: ret i8* %x.addr
}

%ST = type { i8, i8 }

define i8* @test2(%ST* %x, i8* %y) nounwind {
entry:
  %cmp = icmp eq %ST* %x, null
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %incdec.ptr = getelementptr %ST* %x, i32 0, i32 1
  br label %if.end

if.end:
  %x.addr = phi i8* [ %incdec.ptr, %if.then ], [ %y, %entry ]
  ret i8* %x.addr

; CHECK: @test2
; CHECK: %incdec.ptr.y = select i1 %cmp, i8* %incdec.ptr, i8* %y
; CHECK: ret i8* %incdec.ptr.y
}
