/*
http://em386.blogspot.com

You may not use this code in a product,
but feel free to study it and rewrite it
in your own way

This code is an example of how to use the
libelf library for reading ELF objects.

gcc -o libelf-howto libelf-howto.c -lelf
*/

#include "omptarget.h"
#include <cstring>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

std::vector<std::string> g_symbols;
#define ERR -1

Elf *elf;        /* Our Elf pointer for libelf */
Elf_Scn *scn;    /* Section Descriptor */
Elf_Data *edata; /* Data Descriptor */
GElf_Sym sym;    /* Symbol */
GElf_Shdr shdr;  /* Section Header */

struct __tgt_bin_desc binary;
struct __tgt_device_image image;

int main(int argc, char *argv[]) {
  int fd;                // File Descriptor
  char *base_ptr;        // ptr to our object in memory
  char *file = argv[1];  // filename
  struct stat elf_stats; // fstat struct

  if ((fd = open(file, O_RDWR)) == ERR) {
    printf("couldnt open %s\n", file);
    return ERR;
  }

  if ((fstat(fd, &elf_stats))) {
    printf("could not fstat %s\n", file);
    close(fd);
    return ERR;
  }

  if ((base_ptr = (char *)malloc(elf_stats.st_size)) == NULL) {
    printf("could not malloc\n");
    close(fd);
    return ERR;
  }

  if ((read(fd, base_ptr, elf_stats.st_size)) < elf_stats.st_size) {
    printf("could not read %s\n", file);
    free(base_ptr);
    close(fd);
    return ERR;
  }

  /* Check libelf version first */
  if (elf_version(EV_CURRENT) == EV_NONE) {
    printf("WARNING Elf Library is out of date!\n");
  }

  elf = elf_begin(fd, ELF_C_READ,
                  NULL); // Initialize 'elf' pointer to our file descriptor

  int symbol_count;
  int i;

  while ((scn = elf_nextscn(elf, scn)) != NULL) {
    gelf_getshdr(scn, &shdr);

    // When we find a section header marked SHT_SYMTAB stop and get symbols
    if (shdr.sh_type == SHT_SYMTAB) {
      // edata points to our symbol table
      edata = elf_getdata(scn, edata);

      // how many symbols are there? this number comes from the size of
      // the section divided by the entry size
      symbol_count = shdr.sh_size / shdr.sh_entsize;

      // loop through to grab all symbols
      for (i = 0; i < symbol_count; i++) {
        // libelf grabs the symbol data using gelf_getsym()
        gelf_getsym(edata, i, &sym);

        // type of symbol binding
        if (ELF64_ST_BIND(sym.st_info) == STB_GLOBAL) {
          // the name of the symbol is somewhere in a string table
          // we know which one using the shdr.sh_link member
          // libelf grabs the string using elf_strptr()
          char *sym_name = elf_strptr(elf, shdr.sh_link, sym.st_name);

          g_symbols.push_back(std::string(sym_name));
        }
      }
    }
  }

  // construct the omptarget data structures
  image.ImageStart = base_ptr;
  image.ImageEnd = base_ptr + elf_stats.st_size;
  __tgt_offload_entry *entries = (__tgt_offload_entry *)malloc(
      sizeof(__tgt_offload_entry) * g_symbols.size());
  for (int i = 0; i < g_symbols.size(); i++) {
    entries[i].size = 0;
    size_t length = strlen(g_symbols[i].c_str()) + 1;
    entries[i].name = (char *)malloc(length);
    memcpy(entries[i].name, g_symbols[i].c_str(), length);
    entries[i].addr = (void *)entries[i].name;
  }
  image.EntriesBegin = &entries[0];
  image.EntriesEnd = &entries[g_symbols.size()];
  binary.NumDevices = 1;
  binary.DeviceImages = &image;
  binary.EntriesBegin = &entries[0];
  binary.EntriesEnd = &entries[g_symbols.size()];

  // register the library similar to how libomptarget does
  __tgt_register_lib(&binary);

  // setup the kernel arguments
  int device_id = 0;
  void *host_ptr = (void *)entries[0].addr;
#if 1
  const int arg_num = 3;
  const char *in = "Gdkkn\x1FGR@\x1FVnqkc";
  size_t strlength = strlen(in) + 1;
  char *out = (char *)malloc(strlength);

  void *args[] = {(void *)in, out, &strlength};
  void *args_base[] = {(void *)in, out, &strlength};
  int64_t arg_sizes[] = {strlength - 1, strlength - 1, sizeof(size_t)};
  int32_t arg_types[] = {OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_TARGET_PARAM,
                         OMP_TGT_MAPTYPE_FROM | OMP_TGT_MAPTYPE_TARGET_PARAM,
                         OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_TARGET_PARAM};
#else
  const int arg_num = 4;

  int in1[] = {10, 10, 10};
  int in2[] = {20, 20, 20};
  int out[] = {42, 42, 42};
  int N = 3;

  void *args[] = {&N, in1, in2, out};
  void *args_base[] = {&N, in1, in2, out};
  int64_t arg_sizes[] = {sizeof(int), N * sizeof(int), N * sizeof(int),
                         N * sizeof(int)};
  int32_t arg_types[] = {tgt_map_to, tgt_map_to, tgt_map_to, tgt_map_from};
#endif

  // launch
  int x = __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                       arg_types);
#if 1
  out[strlength - 1] = 0;
  printf("-----------------------------------\n");
  printf("Output: %s!\n", out);
  printf("-----------------------------------\n");
#else
  printf("-----------------------------------\n");
  for (int i = 0; i < N; i++)
    printf("Output[%d]: %d!\n", i, out[i]);
  printf("-----------------------------------\n");
#endif
  free(out);
  return 0;
}
