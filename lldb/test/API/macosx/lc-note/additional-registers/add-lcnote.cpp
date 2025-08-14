#include <getopt.h>
#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace std;

[[noreturn]] void print_help(void) {
  fprintf(stderr, "Append an LC_NOTE to a corefile.  Usage: \n");
  fprintf(stderr, "   -i|--input <corefile>\n");
  fprintf(stderr, "   -o|--output <corefile>\n");
  fprintf(stderr, "   -n|--name <LC_NOTE name>\n");
  fprintf(
      stderr,
      "   -r|--remove-dups  remove existing LC_NOTEs with this same name\n");
  fprintf(stderr, "  One of:\n");
  fprintf(stderr, "   -f|--file <file to embed as LC_NOTE payload>\n");
  fprintf(stderr, "   -s|--str <string to embed as LC_NOTE payload>\n");
  exit(1);
}

void parse_args(int argc, char **argv, string &infile, string &outfile,
                string &note_name, vector<uint8_t> &payload,
                bool &remove_dups) {
  const char *const short_opts = "i:o:n:f:s:hr";
  const option long_opts[] = {{"input", required_argument, nullptr, 'i'},
                              {"output", required_argument, nullptr, 'o'},
                              {"name", required_argument, nullptr, 'n'},
                              {"file", required_argument, nullptr, 'f'},
                              {"str", required_argument, nullptr, 's'},
                              {"remove-dups", no_argument, nullptr, 'r'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, no_argument, nullptr, 0}};
  optional<string> infile_str, outfile_str, name_str, payload_file_str,
      payload_str;
  remove_dups = false;
  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (opt == -1)
      break;
    switch (opt) {
    case 'i':
      infile_str = optarg;
      break;
    case 'o':
      outfile_str = optarg;
      break;
    case 'n':
      name_str = optarg;
      break;
    case 'f':
      payload_file_str = optarg;
      break;
    case 's':
      payload_str = optarg;
      break;
    case 'r':
      remove_dups = true;
      break;
    case 'h':
      print_help();
    }
  }

  if (!infile_str || !outfile_str || !name_str ||
      (!payload_file_str && !payload_str))
    print_help();

  infile = *infile_str;
  outfile = *outfile_str;
  note_name = *name_str;
  if (payload_str) {
    payload.resize(payload_str->size(), 0);
    memcpy(payload.data(), payload_str->c_str(), payload_str->size());
  } else {
    struct stat sb;
    if (stat(payload_file_str->c_str(), &sb)) {
      fprintf(stderr, "File '%s' does not exist.\n", payload_file_str->c_str());
      exit(1);
    }
    payload.resize(sb.st_size, 0);
    FILE *f = fopen(payload_file_str->c_str(), "r");
    fread(payload.data(), 1, sb.st_size, f);
    fclose(f);
  }
}

struct all_image_infos_header {
  uint32_t version;         // currently 1
  uint32_t imgcount;        // number of binary images
  uint64_t entries_fileoff; // file offset in the corefile of where the array of
                            // struct entry's begin.
  uint32_t entry_size;      // size of 'struct entry'.
  uint32_t unused;          // set to 0
};

struct image_entry {
  uint64_t filepath_offset; // corefile offset of the c-string filepath,
                            // if available, else this should be set
                            // to UINT64_MAX.
  uuid_t uuid;              // uint8_t[16].  should be set to all zeroes if
                            // uuid is unknown.
  uint64_t
      load_address; // virtual addr of mach-o header, UINT64_MAX if unknown.
  uint64_t seg_addrs_offset; // corefile offset to the array of struct
                             // segment_vmaddr's, UINT64_MAX if none.
  uint32_t segment_count; // The number of segments for this binary, 0 if none.
  uint32_t
      executing; // Set to 0 if executing status is unknown by corefile
                 // creator.
                 // Set to 1 if this binary was executing on any thread,
                 // so it can be force-loaded by the corefile reader.
                 // Set to 2 if this binary was not executing on any thread.
};

int count_lc_notes_with_name(FILE *in, std::string name) {
  fseeko(in, 0, SEEK_SET);

  uint8_t magic[4];
  if (fread(magic, 1, 4, in) != 4) {
    printf("Failed to read magic number\n");
    return 0;
  }
  uint8_t magic_32_le[] = {0xce, 0xfa, 0xed, 0xfe};
  uint8_t magic_64_le[] = {0xcf, 0xfa, 0xed, 0xfe};

  if (memcmp(magic, magic_32_le, 4) != 0 &&
      memcmp(magic, magic_64_le, 4) != 0) {
    return 0;
  }

  fseeko(in, 0, SEEK_SET);

  int number_of_load_cmds = 0;
  size_t size_of_mach_header = 0;
  if (memcmp(magic, magic_64_le, 4) == 0) {
    struct mach_header_64 mh;
    size_of_mach_header = sizeof(mh);
    if (fread(&mh, sizeof(mh), 1, in) != 1) {
      fprintf(stderr, "unable to read mach header\n");
      return 0;
    }
    number_of_load_cmds = mh.ncmds;
  } else {
    struct mach_header mh;
    size_of_mach_header = sizeof(mh);
    if (fread(&mh, sizeof(mh), 1, in) != 1) {
      fprintf(stderr, "unable to read mach header\n");
      return 0;
    }
    number_of_load_cmds = mh.ncmds;
  }

  int notes_seen = 0;
  fseeko(in, size_of_mach_header, SEEK_SET);
  for (int i = 0; i < number_of_load_cmds; i++) {
    off_t cmd_start = ftello(in);
    uint32_t cmd, cmdsize;
    fread(&cmd, sizeof(uint32_t), 1, in);
    fread(&cmdsize, sizeof(uint32_t), 1, in);

    fseeko(in, cmd_start, SEEK_SET);
    off_t next_cmd = cmd_start + cmdsize;
    if (cmd == LC_NOTE) {
      struct note_command note;
      fread(&note, sizeof(note), 1, in);
      if (strncmp(name.c_str(), note.data_owner, 16) == 0)
        notes_seen++;
    }
    fseeko(in, next_cmd, SEEK_SET);
  }
  return notes_seen;
}

void copy_and_add_note(FILE *in, FILE *out, std::string lc_note_name,
                       vector<uint8_t> payload_data, bool remove_dups) {
  int number_of_load_cmds = 0;
  off_t header_start = ftello(in);

  int notes_to_remove = 0;
  if (remove_dups)
    notes_to_remove = count_lc_notes_with_name(in, lc_note_name);
  fseeko(in, header_start, SEEK_SET);

  uint8_t magic[4];
  if (fread(magic, 1, 4, in) != 4) {
    printf("Failed to read magic number\n");
    return;
  }
  uint8_t magic_32_le[] = {0xce, 0xfa, 0xed, 0xfe};
  uint8_t magic_64_le[] = {0xcf, 0xfa, 0xed, 0xfe};

  if (memcmp(magic, magic_32_le, 4) != 0 &&
      memcmp(magic, magic_64_le, 4) != 0) {
    return;
  }

  fseeko(in, header_start, SEEK_SET);

  off_t end_of_infine_loadcmds;
  size_t size_of_mach_header = 0;
  if (memcmp(magic, magic_64_le, 4) == 0) {
    struct mach_header_64 mh;
    size_of_mach_header = sizeof(mh);
    if (fread(&mh, sizeof(mh), 1, in) != 1) {
      fprintf(stderr, "unable to read mach header\n");
      return;
    }
    number_of_load_cmds = mh.ncmds;
    end_of_infine_loadcmds = sizeof(mh) + mh.sizeofcmds;

    mh.ncmds += 1;
    mh.ncmds -= notes_to_remove;
    mh.sizeofcmds += sizeof(struct note_command);
    mh.sizeofcmds -= notes_to_remove * sizeof(struct note_command);
    fseeko(out, header_start, SEEK_SET);
    fwrite(&mh, sizeof(mh), 1, out);
  } else {
    struct mach_header mh;
    size_of_mach_header = sizeof(mh);
    if (fread(&mh, sizeof(mh), 1, in) != 1) {
      fprintf(stderr, "unable to read mach header\n");
      return;
    }
    number_of_load_cmds = mh.ncmds;
    end_of_infine_loadcmds = sizeof(mh) + mh.sizeofcmds;

    mh.ncmds += 1;
    mh.ncmds -= notes_to_remove;
    mh.sizeofcmds += sizeof(struct note_command);
    mh.sizeofcmds -= notes_to_remove * sizeof(struct note_command);
    fseeko(out, header_start, SEEK_SET);
    fwrite(&mh, sizeof(mh), 1, out);
  }

  off_t start_of_infile_load_cmds = ftello(in);
  fseek(in, 0, SEEK_END);
  off_t infile_size = ftello(in);

  // LC_SEGMENT may be aligned to 4k boundaries, let's maintain
  // that alignment by putting 4096 minus the size of the added
  // LC_NOTE load command after the output file's load commands.
  off_t end_of_outfile_loadcmds =
      end_of_infine_loadcmds - (notes_to_remove * sizeof(struct note_command)) +
      4096 - sizeof(struct note_command);
  off_t slide = end_of_outfile_loadcmds - end_of_infine_loadcmds;

  off_t all_image_infos_infile_offset = 0;

  fseek(in, start_of_infile_load_cmds, SEEK_SET);
  fseek(out, start_of_infile_load_cmds, SEEK_SET);
  // Copy all the load commands from IN to OUT, updating any file offsets by
  // SLIDE.
  for (int cmd_num = 0; cmd_num < number_of_load_cmds; cmd_num++) {
    off_t cmd_start = ftello(in);
    uint32_t cmd, cmdsize;
    fread(&cmd, sizeof(uint32_t), 1, in);
    fread(&cmdsize, sizeof(uint32_t), 1, in);

    fseeko(in, cmd_start, SEEK_SET);
    off_t next_cmd = cmd_start + cmdsize;

    switch (cmd) {
    case LC_SEGMENT: {
      struct segment_command segcmd;
      fread(&segcmd, sizeof(segcmd), 1, in);
      segcmd.fileoff += slide;
      fwrite(&segcmd, cmdsize, 1, out);
    } break;
    case LC_SEGMENT_64: {
      struct segment_command_64 segcmd;
      fread(&segcmd, sizeof(segcmd), 1, in);
      segcmd.fileoff += slide;
      fwrite(&segcmd, cmdsize, 1, out);
    } break;
    case LC_NOTE: {
      struct note_command notecmd;
      fread(&notecmd, sizeof(notecmd), 1, in);
      if ((strncmp(lc_note_name.c_str(), notecmd.data_owner, 16) == 0) &&
          remove_dups) {
        fseeko(in, next_cmd, SEEK_SET);
        continue;
      }
      if (strncmp("all image infos", notecmd.data_owner, 16) == 0)
        all_image_infos_infile_offset = notecmd.offset;
      notecmd.offset += slide;
      fwrite(&notecmd, cmdsize, 1, out);
    } break;
    default: {
      vector<uint8_t> buf(cmdsize);
      fread(buf.data(), cmdsize, 1, in);
      fwrite(buf.data(), cmdsize, 1, out);
    }
    }
    fseeko(in, next_cmd, SEEK_SET);
  }

  // Now add our additional LC_NOTE load command.
  struct note_command note;
  note.cmd = LC_NOTE;
  note.cmdsize = sizeof(struct note_command);
  memset(&note.data_owner, 0, 16);
  // data_owner may not be nul terminated if all 16 characters
  // are used, intentionally using strncpy here.
  strncpy(note.data_owner, lc_note_name.c_str(), 16);
  note.offset = infile_size + slide;
  note.size = payload_data.size();
  fwrite(&note, sizeof(struct note_command), 1, out);

  fseeko(in, end_of_infine_loadcmds, SEEK_SET);
  fseeko(out, end_of_outfile_loadcmds, SEEK_SET);

  // Copy the rest of the corefile contents
  vector<uint8_t> data_buf(1024 * 1024);
  while (!feof(in)) {
    size_t read_bytes = fread(data_buf.data(), 1, data_buf.size(), in);
    if (read_bytes > 0) {
      fwrite(data_buf.data(), read_bytes, 1, out);
    } else {
      break;
    }
  }

  fwrite(payload_data.data(), payload_data.size(), 1, out);

  // The "all image infos" LC_NOTE payload has file offsets hardcoded
  // in it, unfortunately.  We've shifted the contents of the corefile
  // and these offsets need to be updated in the ouput file.
  // Re-copy them into the outfile with corrected file offsets.
  off_t infile_image_entry_base = 0;
  if (all_image_infos_infile_offset != 0) {
    off_t all_image_infos_outfile_offset =
        all_image_infos_infile_offset + slide;
    fseeko(in, all_image_infos_infile_offset, SEEK_SET);
    struct all_image_infos_header header;
    fread(&header, sizeof(header), 1, in);
    infile_image_entry_base = header.entries_fileoff;
    header.entries_fileoff += slide;
    fseeko(out, all_image_infos_outfile_offset, SEEK_SET);
    fwrite(&header, sizeof(header), 1, out);

    for (int i = 0; i < header.imgcount; i++) {
      off_t infile_entries_fileoff = header.entries_fileoff - slide;
      off_t outfile_entries_fileoff = header.entries_fileoff;

      struct image_entry ent;
      fseeko(in, infile_entries_fileoff + (header.entry_size * i), SEEK_SET);
      fread(&ent, sizeof(ent), 1, in);
      ent.filepath_offset += slide;
      ent.seg_addrs_offset += slide;
      fseeko(out, outfile_entries_fileoff + (header.entry_size * i), SEEK_SET);
      fwrite(&ent, sizeof(ent), 1, out);
    }
  }
}

int main(int argc, char **argv) {
  string infile, outfile, name;
  vector<uint8_t> payload;
  bool remove_dups;
  parse_args(argc, argv, infile, outfile, name, payload, remove_dups);

  FILE *in = fopen(infile.c_str(), "r");
  if (!in) {
    fprintf(stderr, "Unable to open %s for reading\n", infile.c_str());
    exit(1);
  }
  FILE *out = fopen(outfile.c_str(), "w");
  if (!out) {
    fprintf(stderr, "Unable to open %s for reading\n", outfile.c_str());
    exit(1);
  }

  copy_and_add_note(in, out, name, payload, remove_dups);

  fclose(in);
  fclose(out);
}
