# List all assembler inputs stable test fixtures.
ASM_SOURCES := \
  d_original_example.s \
  regs_int_params.s \
  regs_fp_params.s \
  regs_mixed_params.s \
  live_across_call.s \
  loop_reg_rotate.s \
  seed_reg_const_undef.s

ASM_OBJS := $(ASM_SOURCES:.s=.o)

# Provide a tiny dummy so the harness can link an exe without ASM_OBJS.
C_SOURCES := dummy_main.c

# Generating the dummy source on demand.
dummy_main.c:
	@echo 'int main(void){return 0;}' > $@

# Assemble .s → .o using the configured compiler.
%.o: %.s
	$(CC) -c -x assembler $< -o $@

# Default target: build all .o fixtures and the dummy exe.
.PHONY: all
all: $(ASM_OBJS) $(EXE)

# Keeping things tidy.
clean::
	$(RM) -f $(ASM_OBJS) dummy_main.c

include Makefile.rules
