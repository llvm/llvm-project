; RUN: opt < %s -passes=instcombine -S | grep load | count 3
; PR2471

declare i32 @x(ptr)
define i32 @b(ptr %a, ptr %b) {
entry:
        %tmp1 = load i32, ptr %a            
        %tmp3 = load i32, ptr %b           
        %add = add i32 %tmp1, %tmp3   
        %call = call i32 @x( ptr %a )
        %tobool = icmp ne i32 %add, 0
	; not safe to turn into an uncond load
        %cond = select i1 %tobool, ptr %b, ptr %a             
        %tmp8 = load i32, ptr %cond       
        ret i32 %tmp8
}
