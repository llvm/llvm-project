ARCH = $(shell uname -m)
PROGRAMS = 1lwp_SIGSEGV 2lwp_t2_SIGSEGV 2lwp_process_SIGSEGV
EXECS = $(patsubst %,%.$(ARCH),$(PROGRAMS))
CORES = $(patsubst %,%.core,$(EXECS))

all: $(CORES) $(EXECS)
clean:
	rm -f $(CORES) $(EXECS)

%.core: %
	sysctl -w proc.$$$$.corename=$@; ulimit -s 16; ! ./$<
%.$(ARCH): %.c
	$(CC) -o $@ -g $<

.PHONY: all clean
