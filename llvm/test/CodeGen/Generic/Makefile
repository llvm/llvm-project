# Makefile for running ad-hoc custom LLVM tests
#
%.bc: %.ll
	llvm-as -f $< 
	
%.llc.s: %.bc
	llc -f $< -o $@ 

%.gcc.s: %.c
	gcc -O0 -S $< -o $@

%.nat: %.s
	gcc -O0 -lm $< -o $@

%.cbe.out: %.cbe.nat
	./$< > $@

%.out: %.nat
	./$< > $@

%.clean:
	rm -f $(patsubst %.clean,%.bc,$@) $(patsubst %.clean,%.*.s,$@) \
	      $(patsubst %.clean,%.*.nat,$@) $(patsubst %.clean,%.*.out,$@) 
