; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7

define void @autogen_SD2543() {
A:
  %E83 = add i32 0, 1
  %E820 = add i32 0, undef
  br label %C
C:
  %B908 = add i32 %E83, %E820
  store i32 %B908, i32* undef
  %Sl2391 = select i1 undef, i32 undef, i32 %E83
  %Cmp3114 = icmp ne i32 %Sl2391, undef
  br i1 %Cmp3114, label %C, label %G
G:
  ret void
}
