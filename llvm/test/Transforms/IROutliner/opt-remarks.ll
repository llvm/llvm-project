; RUN: opt -S -verify -iroutliner -o /dev/null \ 
; RUN: -pass-remarks=iroutliner -pass-remarks-missed=iroutliner < %s  \
; RUN: 2>&1 | FileCheck -check-prefix=CHECK %s
; RUN: opt -S -verify -iroutliner -o /dev/null  \
; RUN:  -pass-remarks-output=%t < %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s

; CHECK: remark: <unknown>:0:0: outlined 2 regions with decrease of 31 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION> 
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 5 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 9 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 13 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 17 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 21 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>
; CHECK-NEXT: remark: <unknown>:0:0: did not outline 2 regions due to estimated increase of 7 instructions at locations <UNKNOWN LOCATION> <UNKNOWN LOCATION>

; YAML: --- !Passed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            Outlined
; YAML-NEXT: Function:        function3.outlined
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'outlined '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions with decrease of '
; YAML-NEXT:   - Benefit:         '31'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '5'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '9'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '13'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '17'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '21'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            iroutliner
; YAML-NEXT: Name:            WouldNotDecreaseSize
; YAML-NEXT: Function:        function1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'did not outline '
; YAML-NEXT:   - String:          '2'
; YAML-NEXT:   - String:          ' regions due to estimated increase of '
; YAML-NEXT:   - InstructionIncrease: '7'
; YAML-NEXT:   - String:          ' instructions at locations '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ' '
; YAML-NEXT:   - DebugLoc:        '<UNKNOWN LOCATION>'
; YAML-NEXT: ...

define void @function1() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %output = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 2, i32* %a, align 4
  store i32 3, i32* %b, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add i32 %0, %1
  store i32 %add, i32* %output, align 4
  %2 = load i32, i32* %output, align 4
  %3 = load i32, i32* %output, align 4
  %mul = mul i32 %2, %add
  store i32 %mul, i32* %result, align 4
  ret void
}

define void @function2() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %output = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 2, i32* %a, align 4
  store i32 3, i32* %b, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add i32 %0, %1
  store i32 %add, i32* %output, align 4
  %2 = load i32, i32* %output, align 4
  %mul = mul i32 %2, %add
  store i32 %mul, i32* %result, align 4
  ret void
}

define void @function3() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add i32 %0, %1
  %mul = mul i32 %0, %1
  %sub = sub i32 %0, %1
  %div = sdiv i32 %0, %1
  %add2 = add i32 %0, %1
  %mul2 = mul i32 %0, %1
  %sub2 = sub i32 %0, %1
  %div2 = sdiv i32 %0, %1
  ret void
}

define void @function4() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add i32 %0, %1
  %mul = mul i32 %0, %1
  %sub = sub i32 %0, %1
  %div = sdiv i32 %0, %1
  %add2 = add i32 %0, %1
  %mul2 = mul i32 %0, %1
  %sub2 = sub i32 %0, %1
  %div2 = sdiv i32 %0, %1
  ret void
}
