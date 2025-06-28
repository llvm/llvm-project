# Inline Assembly Safety Directive

## Overview

The `.inline_asm_mode` directive provides enhanced safety for inline assembly blocks by warning about potentially unsafe label usage. This directive helps prevent common errors where programmers create non-local labels in inline assembly that could be inadvertently jumped to from external code.

## Syntax

```assembly
.inline_asm_mode strict   # Enable strict mode - warn on non-local labels
.inline_asm_mode relaxed  # Disable strict mode (default)
```

## Description

When `.inline_asm_mode strict` is active, the assembler will emit warnings for labels that are considered potentially unsafe for inline assembly:

- **Safe labels** (no warnings):
  - Local labels starting with `.L` (e.g., `.L_loop`, `.L_end`)
  - Numeric labels (e.g., `1:`, `42:`)
  - Labels starting with special prefixes (`$`, `__`)
  - Labels starting with `.` (local scope)

- **Unsafe labels** (warnings emitted):
  - Global labels without special prefixes (e.g., `my_function:`, `loop:`)
  - Labels that could be accessed from outside the inline assembly block

## Use Cases

### Frontend Integration

Compiler frontends can use this directive when generating inline assembly:

```c++
// Emitted by the compiler: .inline_asm_mode strict
// C++ inline assembly example
asm(
    ".L_loop:\n"              // Safe - no warning
    "  add %0, %1\n"
    "  jne .L_loop\n"         // Safe - local jump
    "exit:\n"                 // Warning
    : "=r"(result) : "r"(input));
```
// Emitted by the compiler: .inline_asm_mode relaxed

## Rationale

Inline assembly blocks are often embedded within larger functions or modules. Non-local labels in these blocks can create several problems:

1. **Naming conflicts**: Global labels may conflict with other symbols in the compilation unit
2. **Unintended control flow**: External code might accidentally jump to labels intended for internal use
3. **Maintenance issues**: Global labels make inline assembly less encapsulated

The strict mode helps identify these potential issues during compilation, allowing developers to use safer local labels instead.

## Error Handling

Invalid directive usage will produce parse errors:

```assembly
.inline_asm_mode invalid_mode
# Error: expected 'strict' or 'relaxed'

.inline_asm_mode
# Error: expected 'strict' or 'relaxed' after '.inline_asm_mode'
```

## Implementation Details

- The directive affects only subsequent label definitions until changed
- Default mode is `relaxed` (no additional warnings)
- The directive state is maintained in the MC streamer
- Warnings are emitted through the standard LLVM diagnostic system

## Examples

### Complete Example

```assembly
.text
.globl example_function
example_function:
    # Regular function labels (outside inline asm) - no warnings

    # Simulate inline assembly block with safety
    .inline_asm_mode strict

    # These are safe
    .L_inline_start:
        mov $1, %eax
        test %eax, %eax
        jz .L_inline_end

    1:  # Numeric label
        inc %eax
        cmp $10, %eax
        jl 1b

    .L_inline_end:
        # End of safe inline block

    # This would generate a warning
    # global_inline_label:  # Warning would be emitted

    .inline_asm_mode relaxed

    # Back to normal mode
    ret
```

