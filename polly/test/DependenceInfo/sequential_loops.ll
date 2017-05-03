; RUN: opt -S %loadPolly -basicaa -polly-dependences -analyze -polly-dependences-analysis-type=value-based < %s | FileCheck %s -check-prefix=VALUE
; RUN: opt -S %loadPolly -basicaa -polly-dependences -analyze -polly-dependences-analysis-type=memory-based < %s | FileCheck %s -check-prefix=MEMORY
; RUN: opt -S %loadPolly -basicaa -polly-dependences -analyze -polly-dependences-analysis-type=value-based -polly-dependences-analysis-level=access-wise < %s | FileCheck %s -check-prefix=VALUE_ACCESS

; VALUE-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'sequential_writes':
; VALUE-NEXT:      RAW dependences:
; VALUE-NEXT:          {  }
; VALUE-NEXT:      WAR dependences:
; VALUE-NEXT:          {  }
; VALUE-NEXT:      WAW dependences:
; VALUE-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99 }
;
;VALUE_ACCESS-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'sequential_writes':
;VALUE_ACCESS-NEXT:        RAW dependences:
;VALUE_ACCESS-NEXT:                {  }
;VALUE_ACCESS-NEXT:        WAR dependences:
;VALUE_ACCESS-NEXT:                {  }
;VALUE_ACCESS-NEXT:        WAW dependences:
;VALUE_ACCESS-NEXT:                { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Write0[]] -> [Stmt_S2[i0] -> Stmt_S2_Write0[]] : 0 <= i0 <= 9; Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; [Stmt_S2[i0] -> Stmt_S2_Write0[]] -> [Stmt_S3[i0] -> Stmt_S3_Write0[]] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Write0[]] -> [Stmt_S3[i0] -> Stmt_S3_Write0[]] : 10 <= i0 <= 99; Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99 }

;
; VALUE-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'read_after_writes':
; VALUE-NEXT:      RAW dependences:
; VALUE-NEXT:          { Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99 }
; VALUE-NEXT:      WAR dependences:
; VALUE-NEXT:          {  }
; VALUE-NEXT:      WAW dependences:
; VALUE-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9 }
;
;VALUE_ACCESS-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'read_after_writes':
;VALUE_ACCESS-NEXT:        RAW dependences:
;VALUE_ACCESS-NEXT:                { Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99; Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; [Stmt_S2[i0] -> Stmt_S2_Write0[]] -> [Stmt_S3[i0] -> Stmt_S3_Read0[]] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Write0[]] -> [Stmt_S3[i0] -> Stmt_S3_Read0[]] : 10 <= i0 <= 99 }

;VALUE_ACCESS-NEXT:        WAR dependences:
;VALUE_ACCESS-NEXT:                {  }
;VALUE_ACCESS-NEXT:        WAW dependences:
;VALUE_ACCESS-NEXT:                { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Write0[]] -> [Stmt_S2[i0] -> Stmt_S2_Write0[]] : 0 <= i0 <= 9 }
;
; VALUE-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'write_after_read':
; VALUE-NEXT:      RAW dependences:
; VALUE-NEXT:          {  }
; VALUE-NEXT:      WAR dependences:
; VALUE-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99 }
; VALUE-NEXT:      WAW dependences:
; VALUE-NEXT:          { Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9 }
;
;VALUE_ACCESS-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'write_after_read':
;VALUE_ACCESS-NEXT:         RAW dependences:
;VALUE_ACCESS-NEXT:                 {  }
;VALUE_ACCESS-NEXT:         WAR dependences:
;VALUE_ACCESS-NEXT:                { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Read0[]] -> [Stmt_S2[i0] -> Stmt_S2_Write0[]] : 0 <= i0 <= 9; [Stmt_S1[i0] -> Stmt_S1_Read0[]] -> [Stmt_S3[i0] -> Stmt_S3_Write0[]] : 10 <= i0 <= 99; Stmt_S1[i0] -> Stmt_S3[i0] : 10 <= i0 <= 99 }
;VALUE_ACCESS-NEXT:         WAW dependences:
;VALUE_ACCESS-NEXT:                { Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; [Stmt_S2[i0] -> Stmt_S2_Write0[]] -> [Stmt_S3[i0] -> Stmt_S3_Write0[]] : 0 <= i0 <= 9 }
;
; VALUE-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.2' in function 'parametric_offset':
; VALUE-NEXT:      RAW dependences:
; VALUE-NEXT:          [p] -> { Stmt_S1[i0] -> Stmt_S2[-p + i0] : i0 >= p and 0 <= i0 <= 99 and i0 <= 9 + p }
; VALUE-NEXT:      WAR dependences:
; VALUE-NEXT:          [p] -> {  }
; VALUE-NEXT:      WAW dependences:
; VALUE-NEXT:          [p] -> {  }
;
;VALUE_ACCESS-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.2' in function 'parametric_offset':
;VALUE_ACCESS-NEXT:        RAW dependences:
;VALUE_ACCESS-NEXT:                [p] -> { [Stmt_S1[i0] -> Stmt_S1_Write0[]] -> [Stmt_S2[-p + i0] -> Stmt_S2_Read0[]] : i0 >= p and 0 <= i0 <= 99 and i0 <= 9 + p; Stmt_S1[i0] -> Stmt_S2[-p + i0] : i0 >= p and 0 <= i0 <= 99 and i0 <= 9 + p }
;VALUE_ACCESS-NEXT:        WAR dependences:
;VALUE_ACCESS-NEXT:                [p] -> {  }
;VALUE_ACCESS-NEXT:        WAW dependences:
;VALUE_ACCESS-NEXT:                [p] -> {  }

; MEMORY-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'sequential_writes':
; MEMORY-NEXT:      RAW dependences:
; MEMORY-NEXT:          {  }
; MEMORY-NEXT:      WAR dependences:
; MEMORY-NEXT:          {  }
; MEMORY-NEXT:      WAW dependences:
; MEMORY-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 0 <= i0 <= 99 }
;
; MEMORY-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'read_after_writes':
; MEMORY-NEXT:      RAW dependences:
; MEMORY-NEXT:          { Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 0 <= i0 <= 99 }
; MEMORY-NEXT:      WAR dependences:
; MEMORY-NEXT:          {  }
; MEMORY-NEXT:      WAW dependences:
; MEMORY-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9 }
;
; MEMORY-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.3' in function 'write_after_read':
; MEMORY-NEXT:      RAW dependences:
; MEMORY-NEXT:          {  }
; MEMORY-NEXT:      WAR dependences:
; MEMORY-NEXT:          { Stmt_S1[i0] -> Stmt_S2[i0] : 0 <= i0 <= 9; Stmt_S1[i0] -> Stmt_S3[i0] : 0 <= i0 <= 99 }
; MEMORY-NEXT:      WAW dependences:
; MEMORY-NEXT:          { Stmt_S2[i0] -> Stmt_S3[i0] : 0 <= i0 <= 9 }
;
; MEMORY-LABEL: Printing analysis 'Polly - Calculate dependences' for region: 'S1 => exit.2' in function 'parametric_offset':
; MEMORY-NEXT:      RAW dependences:
; MEMORY-NEXT:          [p] -> { Stmt_S1[i0] -> Stmt_S2[-p + i0] : i0 >= p and 0 <= i0 <= 99 and i0 <= 9 + p }
; MEMORY-NEXT:      WAR dependences:
; MEMORY-NEXT:          [p] -> {  }
; MEMORY-NEXT:      WAW dependences:
; MEMORY-NEXT:          [p] -> {  }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

;     for(i = 0; i < 100; i++ )
; S1:   A[i] = 2;
;
;     for (i = 0; i < 10; i++ )
; S2:   A[i]  = 5;
;
;     for (i = 0; i < 200; i++ )
; S3:   A[i] = 5;

define void @sequential_writes() {
entry:
  %A = alloca [200 x i32]
  br label %S1

S1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %S1 ]
  %arrayidx.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  store i32 2, i32* %arrayidx.1
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %S1, label %exit.1

exit.1:
  br label %S2

S2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %S2 ]
  %arrayidx.2 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.2
  store i32 5, i32* %arrayidx.2
  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 10
  br i1 %exitcond.2, label %S2, label %exit.2

exit.2:
  br label %S3

S3:
  %indvar.3 = phi i64 [ 0, %exit.2 ], [ %indvar.next.3, %S3 ]
  %arrayidx.3 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.3
  store i32 7, i32* %arrayidx.3
  %indvar.next.3 = add i64 %indvar.3, 1
  %exitcond.3 = icmp ne i64 %indvar.next.3, 200
  br i1 %exitcond.3, label %S3 , label %exit.3

exit.3:
  ret void
}


;     for(i = 0; i < 100; i++ )
; S1:   A[i] = 2;
;
;     for (i = 0; i < 10; i++ )
; S2:   A[i]  = 5;
;
;     for (i = 0; i < 200; i++ )
; S3:   B[i] = A[i];

define void @read_after_writes() {
entry:
  %A = alloca [200 x i32]
  %B = alloca [200 x i32]
  br label %S1

S1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %S1 ]
  %arrayidx.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  store i32 2, i32* %arrayidx.1
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %S1, label %exit.1

exit.1:
  br label %S2

S2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %S2 ]
  %arrayidx.2 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.2
  store i32 5, i32* %arrayidx.2
  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 10
  br i1 %exitcond.2, label %S2, label %exit.2

exit.2:
  br label %S3

S3:
  %indvar.3 = phi i64 [ 0, %exit.2 ], [ %indvar.next.3, %S3 ]
  %arrayidx.3.a = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.3
  %arrayidx.3.b = getelementptr [200 x i32], [200 x i32]* %B, i64 0, i64 %indvar.3
  %val = load i32, i32* %arrayidx.3.a
  store i32 %val, i32* %arrayidx.3.b
  %indvar.next.3 = add i64 %indvar.3, 1
  %exitcond.3 = icmp ne i64 %indvar.next.3, 200
  br i1 %exitcond.3, label %S3 , label %exit.3

exit.3:
  ret void
}


;     for(i = 0; i < 100; i++ )
; S1:   B[i] = A[i];
;
;     for (i = 0; i < 10; i++ )
; S2:   A[i]  = 5;
;
;     for (i = 0; i < 200; i++ )
; S3:   A[i]  = 10;

define void @write_after_read() {
entry:
  %A = alloca [200 x i32]
  %B = alloca [200 x i32]
  br label %S1

S1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %S1 ]
  %arrayidx.1.a = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  %arrayidx.1.b = getelementptr [200 x i32], [200 x i32]* %B, i64 0, i64 %indvar.1
  %val = load i32, i32* %arrayidx.1.a
  store i32 %val, i32* %arrayidx.1.b
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %S1, label %exit.1

exit.1:
  br label %S2

S2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %S2 ]
  %arrayidx.2 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.2
  store i32 5, i32* %arrayidx.2
  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 10
  br i1 %exitcond.2, label %S2, label %exit.2

exit.2:
  br label %S3

S3:
  %indvar.3 = phi i64 [ 0, %exit.2 ], [ %indvar.next.3, %S3 ]
  %arrayidx.3 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.3
  store i32 10, i32* %arrayidx.3
  %indvar.next.3 = add i64 %indvar.3, 1
  %exitcond.3 = icmp ne i64 %indvar.next.3, 200
  br i1 %exitcond.3, label %S3 , label %exit.3

exit.3:
  ret void
}


;     for(i = 0; i < 100; i++ )
; S1:   A[i] = 10
;
;     for(i = 0; i < 100; i++ )
; S2:   B[i] = A[i + p];

define void @parametric_offset(i64 %p) {
entry:
  %A = alloca [200 x i32]
  %B = alloca [200 x i32]
  br label %S1

S1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %S1 ]
  %arrayidx.1 = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %indvar.1
  store i32 10, i32* %arrayidx.1
  %indvar.next.1 = add i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, 100
  br i1 %exitcond.1, label %S1, label %exit.1

exit.1:
  br label %S2

S2:
  %indvar.2 = phi i64 [ 0, %exit.1 ], [ %indvar.next.2, %S2 ]
  %sum = add i64 %indvar.2, %p
  %arrayidx.2.a = getelementptr [200 x i32], [200 x i32]* %A, i64 0, i64 %sum
  %arrayidx.2.b = getelementptr [200 x i32], [200 x i32]* %B, i64 0, i64 %indvar.2
  %val = load i32, i32* %arrayidx.2.a
  store i32 %val, i32* %arrayidx.2.b
  %indvar.next.2 = add i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, 10
  br i1 %exitcond.2, label %S2, label %exit.2

exit.2:
  ret void
}

