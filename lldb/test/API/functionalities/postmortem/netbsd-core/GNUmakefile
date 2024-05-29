ARCH = $(shell uname -m)
PROGRAMS = 1lwp_SIGSEGV 2lwp_t2_SIGSEGV 2lwp_process_SIGSEGV
EXECS = $(patsubst %,%.$(ARCH),$(PROGRAMS))
CORES = $(patsubst %,%.core,$(EXECS))

all: $(CORES) $(EXECS)
clean:
	rm -f $(CORES) $(EXECS)

%.core: %
	sysctl -w proc.$$$$.corename=$@; ulimit -s 16; ! ./$(call path_wrapper,$<)
%.$(ARCH): %.c
	$(CC) -o $@ -g $(call path_wrapper,$<)

.PHONY: all clean
