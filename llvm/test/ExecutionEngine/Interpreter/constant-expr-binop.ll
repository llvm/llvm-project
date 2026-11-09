; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s

@a = global i32 0
@add = global i64 add (i64 ptrtoint (ptr @a to i64), i64 5)
@sub = global i64 sub (i64 ptrtoint (ptr @a to i64), i64 3)
@xor = global i64 xor (i64 ptrtoint (ptr @a to i64), i64 1)

define i32 @main() {
  %base = ptrtoint ptr @a to i64
  %v.add = load i64, ptr @add
  %exp.add = add i64 %base, 5
  %ok.add = icmp eq i64 %v.add, %exp.add

  %v.sub = load i64, ptr @sub
  %exp.sub = sub i64 %base, 3
  %ok.sub = icmp eq i64 %v.sub, %exp.sub

  %v.xor = load i64, ptr @xor
  %exp.xor = xor i64 %base, 1
  %ok.xor = icmp eq i64 %v.xor, %exp.xor

  %ok.0 = and i1 %ok.add, %ok.sub
  %ok = and i1 %ok.0, %ok.xor
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
