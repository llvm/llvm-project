//===-- ArchiveReader.cpp - Read LLVM archive files -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builds up standard unix archive files (.a) containing LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "ArchiveInternals.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Module.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
using namespace llvm;

/// Read a variable-bit-rate encoded unsigned integer
static inline unsigned readInteger(const char*&At, const char*End) {
  unsigned Shift = 0;
  unsigned Result = 0;

  do {
    if (At == End)
      return Result;
    Result |= (unsigned)((*At++) & 0x7F) << Shift;
    Shift += 7;
  } while (At[-1] & 0x80);
  return Result;
}

// Completely parse the Archive's symbol table and populate symTab member var.
bool
Archive::parseSymbolTable(const void* data, unsigned size, std::string* error) {
  const char* At = (const char*) data;
  const char* End = At + size;
  while (At < End) {
    unsigned offset = readInteger(At, End);
    if (At == End) {
      if (error)
        *error = "Ran out of data reading vbr_uint for symtab offset!";
      return false;
    }
    unsigned length = readInteger(At, End);
    if (At == End) {
      if (error)
        *error = "Ran out of data reading vbr_uint for symtab length!";
      return false;
    }
    if (At + length > End) {
      if (error)
        *error = "Malformed symbol table: length not consistent with size";
      return false;
    }
    // we don't care if it can't be inserted (duplicate entry)
    symTab.insert(std::make_pair(std::string(At, length), offset));
    At += length;
  }
  symTabSize = size;
  return true;
}

// This member parses an ArchiveMemberHeader that is presumed to be pointed to
// by At. The At pointer is updated to the byte just after the header, which
// can be variable in size.
ArchiveMember*
Archive::parseMemberHeader(const char*& At, const char* End, std::string* error)
{
  if (At + sizeof(ArchiveMemberHeader) >= End) {
    if (error)
      *error = "Unexpected end of file";
    return 0;
  }

  // Cast archive member header
  const ArchiveMemberHeader* Hdr = (const ArchiveMemberHeader*)At;
  At += sizeof(ArchiveMemberHeader);

  int flags = 0;
  int MemberSize = atoi(Hdr->size);
  assert(MemberSize >= 0);

  // Check the size of the member for sanity
  if (At + MemberSize > End) {
    if (error)
      *error = "invalid member length in archive file";
    return 0;
  }

  // Check the member signature
  if (!Hdr->checkSignature()) {
    if (error)
      *error = "invalid file member signature";
    return 0;
  }

  // Convert and check the member name
  // The empty name ( '/' and 15 blanks) is for a foreign (non-LLVM) symbol
  // table. The special name "//" and 14 blanks is for a string table, used
  // for long file names. This library doesn't generate either of those but
  // it will accept them. If the name starts with #1/ and the remainder is
  // digits, then those digits specify the length of the name that is
  // stored immediately following the header. The special name
  // __LLVM_SYM_TAB__ identifies the symbol table for LLVM bitcode.
  // Anything else is a regular, short filename that is terminated with
  // a '/' and blanks.

  std::string pathname;
  switch (Hdr->name[0]) {
    case '#':
      if (Hdr->name[1] == '1' && Hdr->name[2] == '/') {
        if (isdigit(Hdr->name[3])) {
          unsigned len = atoi(&Hdr->name[3]);
          const char *nulp = (const char *)memchr(At, '\0', len);
          pathname.assign(At, nulp != 0 ? (uintptr_t)(nulp - At) : len);
          At += len;
          MemberSize -= len;
          flags |= ArchiveMember::HasLongFilenameFlag;
        } else {
          if (error)
            *error = "invalid long filename";
          return 0;
        }
      } else if (Hdr->name[1] == '_' &&
                 (0 == memcmp(Hdr->name, ARFILE_LLVM_SYMTAB_NAME, 16))) {
        // The member is using a long file name (>15 chars) format.
        // This format is standard for 4.4BSD and Mac OSX operating
        // systems. LLVM uses it similarly. In this format, the
        // remainder of the name field (after #1/) specifies the
        // length of the file name which occupy the first bytes of
        // the member's data. The pathname already has the #1/ stripped.
        pathname.assign(ARFILE_LLVM_SYMTAB_NAME);
        flags |= ArchiveMember::LLVMSymbolTableFlag;
      }
      break;
    case '/':
      if (Hdr->name[1]== '/') {
        if (0 == memcmp(Hdr->name, ARFILE_STRTAB_NAME, 16)) {
          pathname.assign(ARFILE_STRTAB_NAME);
          flags |= ArchiveMember::StringTableFlag;
        } else {
          if (error)
            *error = "invalid string table name";
          return 0;
        }
      } else if (Hdr->name[1] == ' ') {
        if (0 == memcmp(Hdr->name, ARFILE_SVR4_SYMTAB_NAME, 16)) {
          pathname.assign(ARFILE_SVR4_SYMTAB_NAME);
          flags |= ArchiveMember::SVR4SymbolTableFlag;
        } else {
          if (error)
            *error = "invalid SVR4 symbol table name";
          return 0;
        }
      } else if (isdigit(Hdr->name[1])) {
        unsigned index = atoi(&Hdr->name[1]);
        if (index < strtab.length()) {
          const char* namep = strtab.c_str() + index;
          const char* endp = strtab.c_str() + strtab.length();
          const char* p = namep;
          const char* last_p = p;
          while (p < endp) {
            if (*p == '\n' && *last_p == '/') {
              pathname.assign(namep, last_p - namep);
              flags |= ArchiveMember::HasLongFilenameFlag;
              break;
            }
            last_p = p;
            p++;
          }
          if (p >= endp) {
            if (error)
              *error = "missing name termiantor in string table";
            return 0;
          }
        } else {
          if (error)
            *error = "name index beyond string table";
          return 0;
        }
      }
      break;
    case '_':
      if (Hdr->name[1] == '_' &&
          (0 == memcmp(Hdr->name, ARFILE_BSD4_SYMTAB_NAME, 16))) {
        pathname.assign(ARFILE_BSD4_SYMTAB_NAME);
        flags |= ArchiveMember::BSD4SymbolTableFlag;
        break;
      }
      /* FALL THROUGH */

    default:
      const char* slash = (const char*) memchr(Hdr->name, '/', 16);
      if (slash == 0)
        slash = Hdr->name + 16;
      pathname.assign(Hdr->name, slash - Hdr->name);
      break;
  }

  // Determine if this is a bitcode file
  switch (sys::IdentifyFileType(At, 4)) {
    case sys::Bitcode_FileType:
      flags |= ArchiveMember::BitcodeFlag;
      break;
    default:
      flags &= ~ArchiveMember::BitcodeFlag;
      break;
  }

  // Instantiate the ArchiveMember to be filled
  ArchiveMember* member = new ArchiveMember(this);

  // Fill in fields of the ArchiveMember
  member->parent = this;
  member->path.set(pathname);
  member->info.fileSize = MemberSize;
  member->info.modTime.fromEpochTime(atoi(Hdr->date));
  unsigned int mode;
  sscanf(Hdr->mode, "%o", &mode);
  member->info.mode = mode;
  member->info.user = atoi(Hdr->uid);
  member->info.group = atoi(Hdr->gid);
  member->flags = flags;
  member->data = At;

  return member;
}

bool
Archive::checkSignature(std::string* error) {
  // Check the magic string at file's header
  if (mapfile->getBufferSize() < 8 || memcmp(base, ARFILE_MAGIC, 8)) {
    if (error)
      *error = "invalid signature for an archive file";
    return false;
  }
  return true;
}

// This function loads the entire archive and fully populates its ilist with
// the members of the archive file. This is typically used in preparation for
// editing the contents of the archive.
bool
Archive::loadArchive(std::string* error) {

  // Set up parsing
  members.clear();
  symTab.clear();
  const char *At = base;
  const char *End = mapfile->getBufferEnd();

  if (!checkSignature(error))
    return false;

  At += 8;  // Skip the magic string.

  bool seenSymbolTable = false;
  bool foundFirstFile = false;
  while (At < End) {
    // parse the member header
    const char* Save = At;
    ArchiveMember* mbr = parseMemberHeader(At, End, error);
    if (!mbr)
      return false;

    // check if this is the foreign symbol table
    if (mbr->isSVR4SymbolTable() || mbr->isBSD4SymbolTable()) {
      // We just save this but don't do anything special
      // with it. It doesn't count as the "first file".
      if (foreignST) {
        // What? Multiple foreign symbol tables? Just chuck it
        // and retain the last one found.
        delete foreignST;
      }
      foreignST = mbr;
      At += mbr->getSize();
      if ((intptr_t(At) & 1) == 1)
        At++;
    } else if (mbr->isStringTable()) {
      // Simply suck the entire string table into a string
      // variable. This will be used to get the names of the
      // members that use the "/ddd" format for their names
      // (SVR4 style long names).
      strtab.assign(At, mbr->getSize());
      At += mbr->getSize();
      if ((intptr_t(At) & 1) == 1)
        At++;
      delete mbr;
    } else if (mbr->isLLVMSymbolTable()) {
      // This is the LLVM symbol table for the archive. If we've seen it
      // already, its an error. Otherwise, parse the symbol table and move on.
      if (seenSymbolTable) {
        if (error)
          *error = "invalid archive: multiple symbol tables";
        return false;
      }
      if (!parseSymbolTable(mbr->getData(), mbr->getSize(), error))
        return false;
      seenSymbolTable = true;
      At += mbr->getSize();
      if ((intptr_t(At) & 1) == 1)
        At++;
      delete mbr; // We don't need this member in the list of members.
    } else {
      // This is just a regular file. If its the first one, save its offset.
      // Otherwise just push it on the list and move on to the next file.
      if (!foundFirstFile) {
        firstFileOffset = Save - base;
        foundFirstFile = true;
      }
      members.push_back(mbr);
      At += mbr->getSize();
      if ((intptr_t(At) & 1) == 1)
        At++;
    }
  }
  return true;
}

// Open and completely load the archive file.
Archive*
Archive::OpenAndLoad(const sys::Path& file, LLVMContext& C, 
                     std::string* ErrorMessage) {
  std::auto_ptr<Archive> result ( new Archive(file, C));
  if (result->mapToMemory(ErrorMessage))
    return 0;
  if (!result->loadArchive(ErrorMessage))
    return 0;
  return result.release();
}

// Get all the bitcode modules from the archive
bool
Archive::getAllModules(std::vector<Module*>& Modules,
                       std::string* ErrMessage) {

  for (iterator I=begin(), E=end(); I != E; ++I) {
    if (I->isBitcode()) {
      std::string FullMemberName = archPath.str() +
        "(" + I->getPath().str() + ")";
      MemoryBuffer *Buffer =
        MemoryBuffer::getMemBufferCopy(StringRef(I->getData(), I->getSize()),
                                       FullMemberName.c_str());
      
      Module *M = ParseBitcodeFile(Buffer, Context, ErrMessage);
      delete Buffer;
      if (!M)
        return true;

      Modules.push_back(M);
    }
  }
  return false;
}

// Load just the symbol table from the archive file
bool
Archive::loadSymbolTable(std::string* ErrorMsg) {

  // Set up parsing
  members.clear();
  symTab.clear();
  const char *At = base;
  const char *End = mapfile->getBufferEnd();

  // Make sure we're dealing with an archive
  if (!checkSignature(ErrorMsg))
    return false;

  At += 8; // Skip signature

  // Parse the first file member header
  const char* FirstFile = At;
  ArchiveMember* mbr = parseMemberHeader(At, End, ErrorMsg);
  if (!mbr)
    return false;

  if (mbr->isSVR4SymbolTable() || mbr->isBSD4SymbolTable()) {
    // Skip the foreign symbol table, we don't do anything with it
    At += mbr->getSize();
    if ((intptr_t(At) & 1) == 1)
      At++;
    delete mbr;

    // Read the next one
    FirstFile = At;
    mbr = parseMemberHeader(At, End, ErrorMsg);
    if (!mbr) {
      delete mbr;
      return false;
    }
  }

  if (mbr->isStringTable()) {
    // Process the string table entry
    strtab.assign((const char*)mbr->getData(), mbr->getSize());
    At += mbr->getSize();
    if ((intptr_t(At) & 1) == 1)
      At++;
    delete mbr;
    // Get the next one
    FirstFile = At;
    mbr = parseMemberHeader(At, End, ErrorMsg);
    if (!mbr) {
      delete mbr;
      return false;
    }
  }

  // See if its the symbol table
  if (mbr->isLLVMSymbolTable()) {
    if (!parseSymbolTable(mbr->getData(), mbr->getSize(), ErrorMsg)) {
      delete mbr;
      return false;
    }

    At += mbr->getSize();
    if ((intptr_t(At) & 1) == 1)
      At++;
    delete mbr;
    // Can't be any more symtab headers so just advance
    FirstFile = At;
  } else {
    // There's no symbol table in the file. We have to rebuild it from scratch
    // because the intent of this method is to get the symbol table loaded so
    // it can be searched efficiently.
    // Add the member to the members list
    members.push_back(mbr);
  }

  firstFileOffset = FirstFile - base;
  return true;
}

// Open the archive and load just the symbol tables
Archive* Archive::OpenAndLoadSymbols(const sys::Path& file,
                                     LLVMContext& C,
                                     std::string* ErrorMessage) {
  std::auto_ptr<Archive> result ( new Archive(file, C) );
  if (result->mapToMemory(ErrorMessage))
    return 0;
  if (!result->loadSymbolTable(ErrorMessage))
    return 0;
  return result.release();
}

// Look up one symbol in the symbol table and return the module that defines
// that symbol.
Module*
Archive::findModuleDefiningSymbol(const std::string& symbol, 
                                  std::string* ErrMsg) {
  SymTabType::iterator SI = symTab.find(symbol);
  if (SI == symTab.end())
    return 0;

  // The symbol table was previously constructed assuming that the members were
  // written without the symbol table header. Because VBR encoding is used, the
  // values could not be adjusted to account for the offset of the symbol table
  // because that could affect the size of the symbol table due to VBR encoding.
  // We now have to account for this by adjusting the offset by the size of the
  // symbol table and its header.
  unsigned fileOffset =
    SI->second +                // offset in symbol-table-less file
    firstFileOffset;            // add offset to first "real" file in archive

  // See if the module is already loaded
  ModuleMap::iterator MI = modules.find(fileOffset);
  if (MI != modules.end())
    return MI->second.first;

  // Module hasn't been loaded yet, we need to load it
  const char* modptr = base + fileOffset;
  ArchiveMember* mbr = parseMemberHeader(modptr, mapfile->getBufferEnd(),
                                         ErrMsg);
  if (!mbr)
    return 0;

  // Now, load the bitcode module to get the Module.
  std::string FullMemberName = archPath.str() + "(" +
    mbr->getPath().str() + ")";
  MemoryBuffer *Buffer =
    MemoryBuffer::getMemBufferCopy(StringRef(mbr->getData(), mbr->getSize()),
                                   FullMemberName.c_str());
  
  Module *m = getLazyBitcodeModule(Buffer, Context, ErrMsg);
  if (!m)
    return 0;

  modules.insert(std::make_pair(fileOffset, std::make_pair(m, mbr)));

  return m;
}

// Look up multiple symbols in the symbol table and return a set of
// Modules that define those symbols.
bool
Archive::findModulesDefiningSymbols(std::set<std::string>& symbols,
                                    SmallVectorImpl<Module*>& result,
                                    std::string* error) {
  if (!mapfile || !base) {
    if (error)
      *error = "Empty archive invalid for finding modules defining symbols";
    return false;
  }

  if (symTab.empty()) {
    // We don't have a symbol table, so we must build it now but lets also
    // make sure that we populate the modules table as we do this to ensure
    // that we don't load them twice when findModuleDefiningSymbol is called
    // below.

    // Get a pointer to the first file
    const char* At  = base + firstFileOffset;
    const char* End = mapfile->getBufferEnd();

    while ( At < End) {
      // Compute the offset to be put in the symbol table
      unsigned offset = At - base - firstFileOffset;

      // Parse the file's header
      ArchiveMember* mbr = parseMemberHeader(At, End, error);
      if (!mbr)
        return false;

      // If it contains symbols
      if (mbr->isBitcode()) {
        // Get the symbols
        std::vector<std::string> symbols;
        std::string FullMemberName = archPath.str() + "(" +
          mbr->getPath().str() + ")";
        Module* M = 
          GetBitcodeSymbols(At, mbr->getSize(), FullMemberName, Context,
                            symbols, error);

        if (M) {
          // Insert the module's symbols into the symbol table
          for (std::vector<std::string>::iterator I = symbols.begin(),
               E=symbols.end(); I != E; ++I ) {
            symTab.insert(std::make_pair(*I, offset));
          }
          // Insert the Module and the ArchiveMember into the table of
          // modules.
          modules.insert(std::make_pair(offset, std::make_pair(M, mbr)));
        } else {
          if (error)
            *error = "Can't parse bitcode member: " + 
              mbr->getPath().str() + ": " + *error;
          delete mbr;
          return false;
        }
      }

      // Go to the next file location
      At += mbr->getSize();
      if ((intptr_t(At) & 1) == 1)
        At++;
    }
  }

  // At this point we have a valid symbol table (one way or another) so we
  // just use it to quickly find the symbols requested.

  SmallPtrSet<Module*, 16> Added;
  for (std::set<std::string>::iterator I=symbols.begin(),
         Next = I,
         E=symbols.end(); I != E; I = Next) {
    // Increment Next before we invalidate it.
    ++Next;

    // See if this symbol exists
    Module* m = findModuleDefiningSymbol(*I,error);
    if (!m)
      continue;
    bool NewMember = Added.insert(m);
    if (!NewMember)
      continue;

    // The symbol exists, insert the Module into our result.
    result.push_back(m);

    // Remove the symbol now that its been resolved.
    symbols.erase(I);
  }
  return true;
}

bool Archive::isBitcodeArchive() {
  // Make sure the symTab has been loaded. In most cases this should have been
  // done when the archive was constructed, but still,  this is just in case.
  if (symTab.empty())
    if (!loadSymbolTable(0))
      return false;

  // Now that we know it's been loaded, return true
  // if it has a size
  if (symTab.size()) return true;

  // We still can't be sure it isn't a bitcode archive
  if (!loadArchive(0))
    return false;

  std::vector<Module *> Modules;
  std::string ErrorMessage;

  // Scan the archive, trying to load a bitcode member.  We only load one to
  // see if this works.
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (!I->isBitcode())
      continue;
    
    std::string FullMemberName = 
      archPath.str() + "(" + I->getPath().str() + ")";

    MemoryBuffer *Buffer =
      MemoryBuffer::getMemBufferCopy(StringRef(I->getData(), I->getSize()),
                                     FullMemberName.c_str());
    Module *M = ParseBitcodeFile(Buffer, Context);
    delete Buffer;
    if (!M)
      return false;  // Couldn't parse bitcode, not a bitcode archive.
    delete M;
    return true;
  }
  
  return false;
}
