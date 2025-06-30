# Inline Assembly Safety Directive

## Overview

The `.inline_asm_mode` directive provides enhanced safety for inline assembly blocks by warning about potentially unsafe label usage. This directive helps prevent common errors where programmers create non-numeric labels in inline assembly that could be inadvertently jumped to from external code.

## Syntax

```assembly
.inline_asm_mode strict   # Enable strict mode - warn on non-numeric labels
.inline_asm_mode relaxed  # Disable strict mode (default)
```

## Description

When `.inline_asm_mode strict` is active, the assembler will emit warnings for labels that are considered potentially unsafe for inline assembly:

- **Safe labels** (no warnings):
  - Numeric labels (e.g., `1:`, `42:`, `999:`)

- **Unsafe labels** (warnings emitted):
  - All non-numeric labels including:
    - Global labels (e.g., `my_function:`, `loop:`)
    - Local labels (e.g., `.L_loop`, `.L_end`)
    - Special prefixed labels (e.g., `$symbol`, `__symbol`)
    - Any label that doesn't start with a digit

## Use Cases

### Frontend Integration

Compiler frontends can use this directive when generating inline assembly:

```c++
// Emitted by the compiler: .inline_asm_mode strict
// C++ inline assembly example
asm(
    "1:\n"                    // Safe - numeric label
    "  add %0, %1\n"
    "  jne 1b\n"              // Safe - numeric jump
    "exit:\n"                 // Warning - non-numeric label
    : "=r"(result) : "r"(input));
```
// Emitted by the compiler: .inline_asm_mode relaxed

### Assembly Development

Assembly programmers can use this directive for safer inline assembly blocks:

```assembly
.inline_asm_mode strict

# Safe labels - no warnings (numeric only)
1:  # Loop start
    inc %eax
    dec %ebx
    jnz 1b      # Jump back to label 1

2:  # Alternative path
    nop
    jmp 3f      # Jump forward to label 3

3:  # End label
    ret

# Unsafe labels - will generate warnings
# unsafe_global:    # Warning: non-numeric label
# .L_local:         # Warning: non-numeric label
# $special:         # Warning: non-numeric label

.inline_asm_mode relaxed
```

## Rationale

Inline assembly blocks are often embedded within larger functions or modules. Non-numeric labels in these blocks can create several problems:

1. **Naming conflicts**: Named labels may conflict with other symbols in the compilation unit
2. **Unintended control flow**: External code might accidentally jump to named labels intended for internal use
3. **Maintenance issues**: Named labels make inline assembly less self-contained
4. **Assembly convention**: Numeric labels are the standard convention for temporary, local labels in assembly

The strict mode helps identify these potential issues during compilation, encouraging developers to use safer numeric labels instead.

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

    # Only numeric labels are safe
    1:  # Safe - numeric label
        mov $1, %eax
        test %eax, %eax
        jz 2f

    3:  # Safe - numeric label
        inc %eax
        cmp $10, %eax
        jl 3b

    2:  # Safe - numeric label
        # End of safe inline block

    # These would generate warnings
    # .L_inline_start:      # Warning - non-numeric
    # global_inline_label:  # Warning - non-numeric

    .inline_asm_mode relaxed

    # Back to normal mode
    ret
```

## Integration with Build Systems

Build systems can use standard LLVM warning controls to manage these diagnostics:

```bash
# Treat inline assembly warnings as errors
llvm-mc -Werror=inline-asm-unsafe-label input.s

# Suppress inline assembly warnings
llvm-mc -Wno-inline-asm-unsafe-label input.s
```

Note: The specific warning flag names are implementation-dependent and may vary.

