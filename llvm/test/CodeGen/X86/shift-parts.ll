; RUN: llc -march=x86-64 < %s | FileCheck %s
; PR4736

%0 = type { i32, i8, [35 x i8] }

@g_144 = external global %0, align 8              ; <%0*> [#uses=1]

; CHECK: shrdq

define i32 @int87(i32 %uint64p_8, i1 %cond) nounwind {
entry:
  %srcval4 = load i320, i320* bitcast (%0* @g_144 to i320*), align 8 ; <i320> [#uses=1]
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3.in.in.in.v = select i1 %cond, i320 192, i320 128 ; <i320> [#uses=1]
  %call3.in.in.in = lshr i320 %srcval4, %call3.in.in.in.v ; <i320> [#uses=1]
  %call3.in = trunc i320 %call3.in.in.in to i32   ; <i32> [#uses=1]
  %tobool = icmp eq i32 %call3.in, 0              ; <i1> [#uses=1]
  br i1 %tobool, label %for.cond, label %if.then

if.then:                                          ; preds = %for.cond
  ret i32 1
}
