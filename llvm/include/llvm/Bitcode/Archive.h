//===-- llvm/Bitcode/Archive.h - LLVM Bitcode Archive -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file declares the Archive and ArchiveMember classes that provide
// manipulation of LLVM Archive files.  The implementation is provided by the
// lib/Bitcode/Archive library.  This library is used to read and write
// archive (*.a) files that contain LLVM bitcode files (or others).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_ARCHIVE_H
#define LLVM_BITCODE_ARCHIVE_H

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/Path.h"
#include <map>
#include <set>

namespace llvm {
  class MemoryBuffer;

// Forward declare classes
class Module;              // From VMCore
class Archive;             // Declared below
class ArchiveMemberHeader; // Internal implementation class
class LLVMContext;         // Global data

/// This class is the main class manipulated by users of the Archive class. It
/// holds information about one member of the Archive. It is also the element
/// stored by the Archive's ilist, the Archive's main abstraction. Because of
/// the special requirements of archive files, users are not permitted to
/// construct ArchiveMember instances. You should obtain them from the methods
/// of the Archive class instead.
/// @brief This class represents a single archive member.
class ArchiveMember : public ilist_node<ArchiveMember> {
  /// @name Types
  /// @{
  public:
    /// These flags are used internally by the archive member to specify various
    /// characteristics of the member. The various "is" methods below provide
    /// access to the flags. The flags are not user settable.
    enum Flags {
      SVR4SymbolTableFlag = 1,     ///< Member is a SVR4 symbol table
      BSD4SymbolTableFlag = 2,     ///< Member is a BSD4 symbol table
      LLVMSymbolTableFlag = 4,     ///< Member is an LLVM symbol table
      BitcodeFlag = 8,             ///< Member is bitcode
      HasPathFlag = 16,            ///< Member has a full or partial path
      HasLongFilenameFlag = 32,    ///< Member uses the long filename syntax
      StringTableFlag = 64         ///< Member is an ar(1) format string table
    };

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// @returns the parent Archive instance
    /// @brief Get the archive associated with this member
    Archive* getArchive() const          { return parent; }

    /// @returns the path to the Archive's file
    /// @brief Get the path to the archive member
    const sys::Path& getPath() const     { return path; }

    /// The "user" is the owner of the file per Unix security. This may not
    /// have any applicability on non-Unix systems but is a required component
    /// of the "ar" file format.
    /// @brief Get the user associated with this archive member.
    unsigned getUser() const             { return info.getUser(); }

    /// The "group" is the owning group of the file per Unix security. This
    /// may not have any applicability on non-Unix systems but is a required
    /// component of the "ar" file format.
    /// @brief Get the group associated with this archive member.
    unsigned getGroup() const            { return info.getGroup(); }

    /// The "mode" specifies the access permissions for the file per Unix
    /// security. This may not have any applicability on non-Unix systems but is
    /// a required component of the "ar" file format.
    /// @brief Get the permission mode associated with this archive member.
    unsigned getMode() const             { return info.getMode(); }

    /// This method returns the time at which the archive member was last
    /// modified when it was not in the archive.
    /// @brief Get the time of last modification of the archive member.
    sys::TimeValue getModTime() const    { return info.getTimestamp(); }

    /// @returns the size of the archive member in bytes.
    /// @brief Get the size of the archive member.
    uint64_t getSize() const             { return info.getSize(); }

    /// This method returns the total size of the archive member as it
    /// appears on disk. This includes the file content, the header, the
    /// long file name if any, and the padding.
    /// @brief Get total on-disk member size.
    unsigned getMemberSize() const;

    /// This method will return a pointer to the in-memory content of the
    /// archive member, if it is available. If the data has not been loaded
    /// into memory, the return value will be null.
    /// @returns a pointer to the member's data.
    /// @brief Get the data content of the archive member
    const char* getData() const { return data; }

    /// @returns true iff the member is a SVR4 (non-LLVM) symbol table
    /// @brief Determine if this member is a SVR4 symbol table.
    bool isSVR4SymbolTable() const { return flags&SVR4SymbolTableFlag; }

    /// @returns true iff the member is a BSD4.4 (non-LLVM) symbol table
    /// @brief Determine if this member is a BSD4.4 symbol table.
    bool isBSD4SymbolTable() const { return flags&BSD4SymbolTableFlag; }

    /// @returns true iff the archive member is the LLVM symbol table
    /// @brief Determine if this member is the LLVM symbol table.
    bool isLLVMSymbolTable() const { return flags&LLVMSymbolTableFlag; }

    /// @returns true iff the archive member is the ar(1) string table
    /// @brief Determine if this member is the ar(1) string table.
    bool isStringTable() const { return flags&StringTableFlag; }

    /// @returns true iff the archive member is a bitcode file.
    /// @brief Determine if this member is a bitcode file.
    bool isBitcode() const { return flags&BitcodeFlag; }

    /// @returns true iff the file name contains a path (directory) component.
    /// @brief Determine if the member has a path
    bool hasPath() const { return flags&HasPathFlag; }

    /// Long filenames are an artifact of the ar(1) file format which allows
    /// up to sixteen characters in its header and doesn't allow a path
    /// separator character (/). To avoid this, a "long format" member name is
    /// allowed that doesn't have this restriction. This method determines if
    /// that "long format" is used for this member.
    /// @returns true iff the file name uses the long form
    /// @brief Determine if the member has a long file name
    bool hasLongFilename() const { return flags&HasLongFilenameFlag; }

    /// This method returns the status info (like Unix stat(2)) for the archive
    /// member. The status info provides the file's size, permissions, and
    /// modification time. The contents of the Path::StatusInfo structure, other
    /// than the size and modification time, may not have utility on non-Unix
    /// systems.
    /// @returns the status info for the archive member
    /// @brief Obtain the status info for the archive member
    const sys::FileStatus &getFileStatus() const { return info; }

    /// This method causes the archive member to be replaced with the contents
    /// of the file specified by \p File. The contents of \p this will be
    /// updated to reflect the new data from \p File. The \p File must exist and
    /// be readable on entry to this method.
    /// @returns true if an error occurred, false otherwise
    /// @brief Replace contents of archive member with a new file.
    bool replaceWith(const sys::Path &aFile, std::string* ErrMsg);

  /// @}
  /// @name Data
  /// @{
  private:
    Archive*            parent;   ///< Pointer to parent archive
    sys::PathWithStatus path;     ///< Path of file containing the member
    sys::FileStatus     info;     ///< Status info (size,mode,date)
    unsigned            flags;    ///< Flags about the archive member
    const char*         data;     ///< Data for the member

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// The default constructor is only used by the Archive's iplist when it
    /// constructs the list's sentry node.
    ArchiveMember();

  private:
    /// Used internally by the Archive class to construct an ArchiveMember.
    /// The contents of the ArchiveMember are filled out by the Archive class.
    explicit ArchiveMember(Archive *PAR);

    // So Archive can construct an ArchiveMember
    friend class llvm::Archive;
  /// @}
};

/// This class defines the interface to LLVM Archive files. The Archive class
/// presents the archive file as an ilist of ArchiveMember objects. The members
/// can be rearranged in any fashion either by directly editing the ilist or by
/// using editing methods on the Archive class (recommended). The Archive
/// class also provides several ways of accessing the archive file for various
/// purposes such as editing and linking.  Full symbol table support is provided
/// for loading only those files that resolve symbols. Note that read
/// performance of this library is _crucial_ for performance of JIT type
/// applications and the linkers. Consequently, the implementation of the class
/// is optimized for reading.
class Archive {

  /// @name Types
  /// @{
  public:
    /// This is the ilist type over which users may iterate to examine
    /// the contents of the archive
    /// @brief The ilist type of ArchiveMembers that Archive contains.
    typedef iplist<ArchiveMember> MembersList;

    /// @brief Forward mutable iterator over ArchiveMember
    typedef MembersList::iterator iterator;

    /// @brief Forward immutable iterator over ArchiveMember
    typedef MembersList::const_iterator const_iterator;

    /// @brief Reverse mutable iterator over ArchiveMember
    typedef std::reverse_iterator<iterator> reverse_iterator;

    /// @brief Reverse immutable iterator over ArchiveMember
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    /// @brief The in-memory version of the symbol table
    typedef std::map<std::string,unsigned> SymTabType;

  /// @}
  /// @name ilist accessor methods
  /// @{
  public:
    inline iterator               begin()        { return members.begin();  }
    inline const_iterator         begin()  const { return members.begin();  }
    inline iterator               end  ()        { return members.end();    }
    inline const_iterator         end  ()  const { return members.end();    }

    inline reverse_iterator       rbegin()       { return members.rbegin(); }
    inline const_reverse_iterator rbegin() const { return members.rbegin(); }
    inline reverse_iterator       rend  ()       { return members.rend();   }
    inline const_reverse_iterator rend  () const { return members.rend();   }

    inline size_t                 size()   const { return members.size();   }
    inline bool                   empty()  const { return members.empty();  }
    inline const ArchiveMember&   front()  const { return members.front();  }
    inline       ArchiveMember&   front()        { return members.front();  }
    inline const ArchiveMember&   back()   const { return members.back();   }
    inline       ArchiveMember&   back()         { return members.back();   }

  /// @}
  /// @name ilist mutator methods
  /// @{
  public:
    /// This method splices a \p src member from an archive (possibly \p this),
    /// to a position just before the member given by \p dest in \p this. When
    /// the archive is written, \p src will be written in its new location.
    /// @brief Move a member to a new location
    inline void splice(iterator dest, Archive& arch, iterator src)
      { return members.splice(dest,arch.members,src); }

    /// This method erases a \p target member from the archive. When the
    /// archive is written, it will no longer contain \p target. The associated
    /// ArchiveMember is deleted.
    /// @brief Erase a member.
    inline iterator erase(iterator target) { return members.erase(target); }

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Create an empty archive file and associate it with the \p Filename. This
    /// method does not actually create the archive disk file. It creates an
    /// empty Archive object. If the writeToDisk method is called, the archive
    /// file \p Filename will be created at that point, with whatever content
    /// the returned Archive object has at that time.
    /// @returns An Archive* that represents the new archive file.
    /// @brief Create an empty Archive.
    static Archive* CreateEmpty(
      const sys::Path& Filename,///< Name of the archive to (eventually) create.
      LLVMContext& C            ///< Context to use for global information
    );

    /// Open an existing archive and load its contents in preparation for
    /// editing. After this call, the member ilist is completely populated based
    /// on the contents of the archive file. You should use this form of open if
    /// you intend to modify the archive or traverse its contents (e.g. for
    /// printing).
    /// @brief Open and load an archive file
    static Archive* OpenAndLoad(
      const sys::Path& filePath,  ///< The file path to open and load
      LLVMContext& C,       ///< The context to use for global information
      std::string* ErrorMessage   ///< An optional error string
    );

    /// This method opens an existing archive file from \p Filename and reads in
    /// its symbol table without reading in any of the archive's members. This
    /// reduces both I/O and cpu time in opening the archive if it is to be used
    /// solely for symbol lookup (e.g. during linking).  The \p Filename must
    /// exist and be an archive file or an error will be returned. This form
    /// of opening the archive is intended for read-only operations that need to
    /// locate members via the symbol table for link editing.  Since the archve
    /// members are not read by this method, the archive will appear empty upon
    /// return. If editing operations are performed on the archive, they will
    /// completely replace the contents of the archive! It is recommended that
    /// if this form of opening the archive is used that only the symbol table
    /// lookup methods (getSymbolTable, findModuleDefiningSymbol, and
    /// findModulesDefiningSymbols) be used.
    /// @returns an Archive* that represents the archive file, or null on error.
    /// @brief Open an existing archive and load its symbols.
    static Archive* OpenAndLoadSymbols(
      const sys::Path& Filename,   ///< Name of the archive file to open
      LLVMContext& C,              ///< The context to use for global info
      std::string* ErrorMessage=0  ///< An optional error string
    );

    /// This destructor cleans up the Archive object, releases all memory, and
    /// closes files. It does nothing with the archive file on disk. If you
    /// haven't used the writeToDisk method by the time the destructor is
    /// called, all changes to the archive will be lost.
    /// @brief Destruct in-memory archive
    ~Archive();

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// @returns the path to the archive file.
    /// @brief Get the archive path.
    const sys::Path& getPath() { return archPath; }

    /// This method is provided so that editing methods can be invoked directly
    /// on the Archive's iplist of ArchiveMember. However, it is recommended
    /// that the usual STL style iterator interface be used instead.
    /// @returns the iplist of ArchiveMember
    /// @brief Get the iplist of the members
    MembersList& getMembers() { return members; }

    /// This method allows direct query of the Archive's symbol table. The
    /// symbol table is a std::map of std::string (the symbol) to unsigned (the
    /// file offset). Note that for efficiency reasons, the offset stored in
    /// the symbol table is not the actual offset. It is the offset from the
    /// beginning of the first "real" file member (after the symbol table). Use
    /// the getFirstFileOffset() to obtain that offset and add this value to the
    /// offset in the symbol table to obtain the real file offset. Note that
    /// there is purposefully no interface provided by Archive to look up
    /// members by their offset. Use the findModulesDefiningSymbols and
    /// findModuleDefiningSymbol methods instead.
    /// @returns the Archive's symbol table.
    /// @brief Get the archive's symbol table
    const SymTabType& getSymbolTable() { return symTab; }

    /// This method returns the offset in the archive file to the first "real"
    /// file member. Archive files, on disk, have a signature and might have a
    /// symbol table that precedes the first actual file member. This method
    /// allows you to determine what the size of those fields are.
    /// @returns the offset to the first "real" file member  in the archive.
    /// @brief Get the offset to the first "real" file member  in the archive.
    unsigned getFirstFileOffset() { return firstFileOffset; }

    /// This method will scan the archive for bitcode modules, interpret them
    /// and return a vector of the instantiated modules in \p Modules. If an
    /// error occurs, this method will return true. If \p ErrMessage is not null
    /// and an error occurs, \p *ErrMessage will be set to a string explaining
    /// the error that occurred.
    /// @returns true if an error occurred
    /// @brief Instantiate all the bitcode modules located in the archive
    bool getAllModules(std::vector<Module*>& Modules, std::string* ErrMessage);

    /// This accessor looks up the \p symbol in the archive's symbol table and
    /// returns the associated module that defines that symbol. This method can
    /// be called as many times as necessary. This is handy for linking the
    /// archive into another module based on unresolved symbols. Note that the
    /// Module returned by this accessor should not be deleted by the caller. It
    /// is managed internally by the Archive class. It is possible that multiple
    /// calls to this accessor will return the same Module instance because the
    /// associated module defines multiple symbols.
    /// @returns The Module* found or null if the archive does not contain a
    /// module that defines the \p symbol.
    /// @brief Look up a module by symbol name.
    Module* findModuleDefiningSymbol(
      const std::string& symbol,  ///< Symbol to be sought
      std::string* ErrMessage     ///< Error message storage, if non-zero
    );

    /// This method is similar to findModuleDefiningSymbol but allows lookup of
    /// more than one symbol at a time. If \p symbols contains a list of
    /// undefined symbols in some module, then calling this method is like
    /// making one complete pass through the archive to resolve symbols but is
    /// more efficient than looking at the individual members. Note that on
    /// exit, the symbols resolved by this method will be removed from \p
    /// symbols to ensure they are not re-searched on a subsequent call. If
    /// you need to retain the list of symbols, make a copy.
    /// @brief Look up multiple symbols in the archive.
    bool findModulesDefiningSymbols(
      std::set<std::string>& symbols,     ///< Symbols to be sought
      SmallVectorImpl<Module*>& modules,  ///< The modules matching \p symbols
      std::string* ErrMessage             ///< Error msg storage, if non-zero
    );

    /// This method determines whether the archive is a properly formed llvm
    /// bitcode archive.  It first makes sure the symbol table has been loaded
    /// and has a non-zero size.  If it does, then it is an archive.  If not,
    /// then it tries to load all the bitcode modules of the archive.  Finally,
    /// it returns whether it was successful.
    /// @returns true if the archive is a proper llvm bitcode archive
    /// @brief Determine whether the archive is a proper llvm bitcode archive.
    bool isBitcodeArchive();

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// This method is the only way to get the archive written to disk. It
    /// creates or overwrites the file specified when \p this was created
    /// or opened. The arguments provide options for writing the archive. If
    /// \p CreateSymbolTable is true, the archive is scanned for bitcode files
    /// and a symbol table of the externally visible function and global
    /// variable names is created. If \p TruncateNames is true, the names of the
    /// archive members will have their path component stripped and the file
    /// name will be truncated at 15 characters. If \p Compress is specified,
    /// all archive members will be compressed before being written. If
    /// \p PrintSymTab is true, the symbol table will be printed to std::cout.
    /// @returns true if an error occurred, \p error set to error message;
    /// returns false if the writing succeeded.
    /// @brief Write (possibly modified) archive contents to disk
    bool writeToDisk(
      bool CreateSymbolTable=false,   ///< Create Symbol table
      bool TruncateNames=false,       ///< Truncate the filename to 15 chars
      std::string* ErrMessage=0       ///< If non-null, where error msg is set
    );

    /// This method adds a new file to the archive. The \p filename is examined
    /// to determine just enough information to create an ArchiveMember object
    /// which is then inserted into the Archive object's ilist at the location
    /// given by \p where.
    /// @returns true if an error occurred, false otherwise
    /// @brief Add a file to the archive.
    bool addFileBefore(
      const sys::Path& filename, ///< The file to be added
      iterator where,            ///< Insertion point
      std::string* ErrMsg        ///< Optional error message location
    );

  /// @}
  /// @name Implementation
  /// @{
  protected:
    /// @brief Construct an Archive for \p filename and optionally  map it
    /// into memory.
    explicit Archive(const sys::Path& filename, LLVMContext& C);

    /// @param data The symbol table data to be parsed
    /// @param len  The length of the symbol table data
    /// @param error Set to address of a std::string to get error messages
    /// @returns false on error
    /// @brief Parse the symbol table at \p data.
    bool parseSymbolTable(const void* data,unsigned len,std::string* error);

    /// @returns A fully populated ArchiveMember or 0 if an error occurred.
    /// @brief Parse the header of a member starting at \p At
    ArchiveMember* parseMemberHeader(
      const char*&At,    ///< The pointer to the location we're parsing
      const char*End,    ///< The pointer to the end of the archive
      std::string* error ///< Optional error message catcher
    );

    /// @param ErrMessage Set to address of a std::string to get error messages
    /// @returns false on error
    /// @brief Check that the archive signature is correct
    bool checkSignature(std::string* ErrMessage);

    /// @param ErrMessage Set to address of a std::string to get error messages
    /// @returns false on error
    /// @brief Load the entire archive.
    bool loadArchive(std::string* ErrMessage);

    /// @param ErrMessage Set to address of a std::string to get error messages
    /// @returns false on error
    /// @brief Load just the symbol table.
    bool loadSymbolTable(std::string* ErrMessage);

    /// @brief Write the symbol table to an ofstream.
    void writeSymbolTable(std::ofstream& ARFile);

    /// Writes one ArchiveMember to an ofstream. If an error occurs, returns
    /// false, otherwise true. If an error occurs and error is non-null then
    /// it will be set to an error message.
    /// @returns false if writing member succeeded,
    /// returns true if writing member failed, \p error set to error message.
    bool writeMember(
      const ArchiveMember& member, ///< The member to be written
      std::ofstream& ARFile,       ///< The file to write member onto
      bool CreateSymbolTable,      ///< Should symbol table be created?
      bool TruncateNames,          ///< Should names be truncated to 11 chars?
      std::string* ErrMessage      ///< If non-null, place were error msg is set
    );

    /// @brief Fill in an ArchiveMemberHeader from ArchiveMember.
    bool fillHeader(const ArchiveMember&mbr,
                    ArchiveMemberHeader& hdr,int sz, bool TruncateNames) const;

    /// @brief Maps archive into memory
    bool mapToMemory(std::string* ErrMsg);

    /// @brief Frees all the members and unmaps the archive file.
    void cleanUpMemory();

    /// This type is used to keep track of bitcode modules loaded from the
    /// symbol table. It maps the file offset to a pair that consists of the
    /// associated ArchiveMember and the Module.
    /// @brief Module mapping type
    typedef std::map<unsigned,std::pair<Module*,ArchiveMember*> >
      ModuleMap;


  /// @}
  /// @name Data
  /// @{
  protected:
    sys::Path archPath;       ///< Path to the archive file we read/write
    MembersList members;      ///< The ilist of ArchiveMember
    MemoryBuffer *mapfile;    ///< Raw Archive contents mapped into memory
    const char* base;         ///< Base of the memory mapped file data
    SymTabType symTab;        ///< The symbol table
    std::string strtab;       ///< The string table for long file names
    unsigned symTabSize;      ///< Size in bytes of symbol table
    unsigned firstFileOffset; ///< Offset to first normal file.
    ModuleMap modules;        ///< The modules loaded via symbol lookup.
    ArchiveMember* foreignST; ///< This holds the foreign symbol table.
    LLVMContext& Context;     ///< This holds global data.
  /// @}
  /// @name Hidden
  /// @{
  private:
    Archive() LLVM_DELETED_FUNCTION;
    Archive(const Archive&) LLVM_DELETED_FUNCTION;
    Archive& operator=(const Archive&) LLVM_DELETED_FUNCTION;
  /// @}
};

} // End llvm namespace

#endif
