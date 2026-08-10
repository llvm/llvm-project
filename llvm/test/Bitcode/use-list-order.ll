; RUN: verify-uselistorder < %s

@a = global [4 x i1] [i1 0, i1 1, i1 0, i1 1]
@b = alias i1, getelementptr ([4 x i1], [4 x i1]* @a, i64 0, i64 2)

; Check use-list order of constants used by globals.
@glob1 = global i5 7
@glob2 = global i5 7
@glob3 = global i5 7

; Check use-list order between variables and aliases.
@target = global i3 zeroinitializer
@alias1 = alias i3, i3* @target
@alias2 = alias i3, i3* @target
@alias3 = alias i3, i3* @target
@var1 = global i3* @target
@var2 = global i3* @target
@var3 = global i3* @target

; Check use-list order for a global when used both by a global and in a
; function.
@globalAndFunction = global i4 4
@globalAndFunctionGlobalUser = global i4* @globalAndFunction

; Check use-list order for constants used by globals that are themselves used
; as aliases.  This confirms that this globals are recognized as GlobalValues
; (not general constants).
@const.global = global i63 0
@const.global.ptr = global i63* @const.global
@const.global.2 = global i63 0

; Same as above, but for aliases.
@const.target = global i62 1
@const.alias = alias i62, i62* @const.target
@const.alias.ptr = alias i62, i62* @const.alias
@const.alias.2 = alias i62, i62* @const.target

define i64 @f(i64 %f) {
entry:
  %sum = add i64 %f, 0
  ret i64 %sum
}

define i64 @g(i64 %g) {
entry:
  %sum = add i64 %g, 0
  ret i64 %sum
}

define i64 @h(i64 %h) {
entry:
  %sum = add i64 %h, 0
  ret i64 %sum
}

define i64 @i(i64 %i) {
entry:
  %sum = add i64 %i, 1
  ret i64 %sum
}

define i64 @j(i64 %j) {
entry:
  %sum = add i64 %j, 1
  ret i64 %sum
}

define i64 @k(i64 %k) {
entry:
  %sum = add i64 %k, 1
  ret i64 %sum
}

define i64 @l(i64 %l) {
entry:
  %sum = add i64 %l, 1
  ret i64 %sum
}

define i1 @loadb() {
entry:
  %b = load i1, i1* @b
  ret i1 %b
}

define i1 @loada() {
entry:
  %a = load i1, i1* getelementptr ([4 x i1], [4 x i1]* @a, i64 0, i64 2)
  ret i1 %a
}

define i32 @f32(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  br label %first

second:
  %eh = mul i32 %e, %h
  %sum = add i32 %eh, %ef
  br label %exit

exit:
  %product = phi i32 [%ef, %first], [%sum, %second]
  ret i32 %product

first:
  %e = add i32 %a, 7
  %f = add i32 %b, 7
  %g = add i32 %c, 8
  %h = add i32 %d, 8
  %ef = mul i32 %e, %f
  %gh = mul i32 %g, %h
  %gotosecond = icmp slt i32 %gh, -9
  br i1 %gotosecond, label %second, label %exit
}

define i4 @globalAndFunctionFunctionUser() {
entry:
  %local = load i4, i4* @globalAndFunction
  ret i4 %local
}

; Check for when an instruction is its own user.
define void @selfUser(i1 %a) {
entry:
  ret void

loop1:
  br label %loop2

loop2:
  %var = phi i32 [ %var, %loop1 ], [ %var, %loop2 ]
  br label %loop2
}

; Check that block addresses work.
@ba1 = constant i8* blockaddress (@bafunc1, %bb)
@ba2 = constant i8* getelementptr (i8, i8* blockaddress (@bafunc2, %bb), i61 0)
@ba3 = constant i8* getelementptr (i8, i8* blockaddress (@bafunc2, %bb), i61 0)

define i8* @babefore() {
  ret i8* getelementptr (i8, i8* blockaddress (@bafunc2, %bb), i61 0)
bb1:
  ret i8* blockaddress (@bafunc1, %bb)
bb2:
  ret i8* blockaddress (@bafunc3, %bb)
}
define void @bafunc1() {
  unreachable
bb:
  unreachable
}
define void @bafunc2() {
  unreachable
bb:
  unreachable
}
define void @bafunc3() {
  unreachable
bb:
  unreachable
}
define i8* @baafter() {
  ret i8* blockaddress (@bafunc2, %bb)
bb1:
  ret i8* blockaddress (@bafunc1, %bb)
bb2:
  ret i8* blockaddress (@bafunc3, %bb)
}
