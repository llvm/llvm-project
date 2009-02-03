; RUN: llvm-as %s -o - | lli -force-interpreter

target datalayout = "e"

define i32 @main() {
entry:
	%i = alloca i93
	store i93 18364758544493064720, i93* %i
	%i1 = load i93* %i
	%i2 = shl i93 %i1, 32
	%i3 = or i93 %i2, 3753679480
	store i93 %i3, i93* %i
	%i4 = load i93* %i
	%C = icmp eq i93 %i3, %i4
	br i1 %C, label %ok1, label %fail
ok1:
	%b = bitcast i93* %i to [12 x i8]*
        %b0 = getelementptr [12 x i8]* %b, i32 0, i32 0
	%v0 = load i8* %b0
        %c0 = icmp eq i8 %v0, 120
	br i1 %c0, label %ok2, label %fail
ok2:
        %b1 = getelementptr [12 x i8]* %b, i32 0, i32 1
	%v1 = load i8* %b1
        %c1 = icmp eq i8 %v1, 154
	br i1 %c1, label %ok3, label %fail
ok3:
        %b2 = getelementptr [12 x i8]* %b, i32 0, i32 2
	%v2 = load i8* %b2
        %c2 = icmp eq i8 %v2, 188
	br i1 %c2, label %ok4, label %fail
ok4:
        %b3 = getelementptr [12 x i8]* %b, i32 0, i32 3
	%v3 = load i8* %b3
        %c3 = icmp eq i8 %v3, 223
	br i1 %c3, label %ok5, label %fail
ok5:
        %b4 = getelementptr [12 x i8]* %b, i32 0, i32 4
	%v4 = load i8* %b4
        %c4 = icmp eq i8 %v4, 16
	br i1 %c4, label %ok6, label %fail
ok6:
        %b5 = getelementptr [12 x i8]* %b, i32 0, i32 5
	%v5 = load i8* %b5
        %c5 = icmp eq i8 %v5, 50
	br i1 %c5, label %ok7, label %fail
ok7:
        %b6 = getelementptr [12 x i8]* %b, i32 0, i32 6
	%v6 = load i8* %b6
        %c6 = icmp eq i8 %v6, 84
	br i1 %c6, label %ok8, label %fail
ok8:
        %b7 = getelementptr [12 x i8]* %b, i32 0, i32 7
	%v7 = load i8* %b7
        %c7 = icmp eq i8 %v7, 118
	br i1 %c7, label %ok9, label %fail
ok9:
        %b8 = getelementptr [12 x i8]* %b, i32 0, i32 8
	%v8 = load i8* %b8
        %c8 = icmp eq i8 %v8, 152
	br i1 %c8, label %okA, label %fail
okA:
        %b9 = getelementptr [12 x i8]* %b, i32 0, i32 9
	%v9 = load i8* %b9
        %c9 = icmp eq i8 %v9, 186
	br i1 %c9, label %okB, label %fail
okB:
        %bA = getelementptr [12 x i8]* %b, i32 0, i32 10
	%vA = load i8* %bA
        %cA = icmp eq i8 %vA, 220
	br i1 %cA, label %okC, label %fail
okC:
        %bB = getelementptr [12 x i8]* %b, i32 0, i32 11
	%vB = load i8* %bB
        %cB = icmp eq i8 %vB, 30
	br i1 %cB, label %okD, label %fail
okD:
        ret i32 0
fail:
	ret i32 1
}
