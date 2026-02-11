C_SOURCES := main.c file_with_zcd.c file_with_zcmp.c

include Makefile.rules

CFLAGS_NO_ARCH := $(filter-out -march=%,$(CFLAGS))

file_with_zcd.o: file_with_zcd.c
	$(CC) $(CFLAGS_NO_ARCH) -march=rv64gc -g -c $< -o $@

file_with_zcmp.o: file_with_zcmp.c
	$(CC) $(CFLAGS_NO_ARCH) -march=rv64imad_zcmp -g -c $< -o $@

main.o: main.c
	$(CC) $(CFLAGS_NO_ARCH) -march=rv64gc -g -c $< -o $@

$(EXE): main.o file_with_zcd.o file_with_zcmp.o
	$(CC) $^ $(LDFLAGS) -o $@
