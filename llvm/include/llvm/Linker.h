//===- llvm/Linker.h - Module Linker Interface ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface to the module/file/archive linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_H
#define LLVM_LINKER_H

#include "llvm/System/Path.h"
#include <memory>
#include <vector>

namespace llvm {

class Module;

/// This class provides the core functionality of linking in LLVM. It retains a
/// Module object which is the composite of the modules and libraries linked
/// into it. The composite Module can be retrieved via the getModule() method.
/// In this case the Linker still retains ownership of the Module. If the
/// releaseModule() method is used, the ownership of the Module is transferred
/// to the caller and the Linker object is only suitable for destruction.
/// The Linker can link Modules from memory, bitcode files, or bitcode
/// archives.  It retains a set of search paths in which to find any libraries
/// presented to it. By default, the linker will generate error and warning
/// messages to std::cerr but this capability can be turned off with the
/// QuietWarnings and QuietErrors flags. It can also be instructed to verbosely
/// print out the linking actions it is taking with the Verbose flag.
/// @brief The LLVM Linker.
class Linker {

  /// @name Types
  /// @{
  public:
    /// This type is used to pass the linkage items (libraries and files) to
    /// the LinkItems function. It is composed of string/bool pairs. The string
    /// provides the name of the file or library (as with the -l option). The
    /// bool should be true for libraries and false for files, signifying
    /// "isLibrary".
    /// @brief A list of linkage items
    typedef std::vector<std::pair<std::string,bool> > ItemList;

    /// This enumeration is used to control various optional features of the
    /// linker.
    enum ControlFlags {
      Verbose       = 1, ///< Print to std::cerr what steps the linker is taking
      QuietWarnings = 2, ///< Don't print warnings to std::cerr.
      QuietErrors   = 4  ///< Don't print errors to std::cerr.
    };

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Construct the Linker with an empty module which will be given the
    /// name \p progname. \p progname will also be used for error messages.
    /// @brief Construct with empty module
    Linker(
        const std::string& progname, ///< name of tool running linker
        const std::string& modulename, ///< name of linker's end-result module
        unsigned Flags = 0  ///< ControlFlags (one or more |'d together)
    );

    /// Construct the Linker with a previously defined module, \p aModule. Use
    /// \p progname for the name of the program in error messages.
    /// @brief Construct with existing module
    Linker(const std::string& progname, Module* aModule, unsigned Flags = 0);

    /// Destruct the Linker.
    /// @brief Destructor
    ~Linker();

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// This method gets the composite module into which linking is being
    /// done. The Composite module starts out empty and accumulates modules
    /// linked into it via the various LinkIn* methods. This method does not
    /// release the Module to the caller. The Linker retains ownership and will
    /// destruct the Module when the Linker is destructed.
    /// @see releaseModule
    /// @brief Get the linked/composite module.
    Module* getModule() const { return Composite; }

    /// This method releases the composite Module into which linking is being
    /// done. Ownership of the composite Module is transferred to the caller who
    /// must arrange for its destruct. After this method is called, the Linker
    /// terminates the linking session for the returned Module. It will no
    /// longer utilize the returned Module but instead resets itself for
    /// subsequent linking as if the constructor had been called. The Linker's
    /// LibPaths and flags to be reset, and memory will be released.
    /// @brief Release the linked/composite module.
    Module* releaseModule();

    /// This method gets the list of libraries that form the path that the
    /// Linker will search when it is presented with a library name.
    /// @brief Get the Linkers library path
    const std::vector<sys::Path>& getLibPaths() const { return LibPaths; }

    /// This method returns an error string suitable for printing to the user.
    /// The return value will be empty unless an error occurred in one of the
    /// LinkIn* methods. In those cases, the LinkIn* methods will have returned
    /// true, indicating an error occurred. At most one error is retained so
    /// this function always returns the last error that occurred. Note that if
    /// the Quiet control flag is not set, the error string will have already
    /// been printed to std::cerr.
    /// @brief Get the text of the last error that occurred.
    const std::string& getLastError() const { return Error; }

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// Add a path to the list of paths that the Linker will search. The Linker
    /// accumulates the set of libraries added
    /// library paths for the target platform. The standard libraries will
    /// always be searched last. The added libraries will be searched in the
    /// order added.
    /// @brief Add a path.
    void addPath(const sys::Path& path);

    /// Add a set of paths to the list of paths that the linker will search. The
    /// Linker accumulates the set of libraries added. The \p paths will be
    /// added to the end of the Linker's list. Order will be retained.
    /// @brief Add a set of paths.
    void addPaths(const std::vector<std::string>& paths);

    /// This method augments the Linker's list of library paths with the system
    /// paths of the host operating system, include LLVM_LIB_SEARCH_PATH.
    /// @brief Add the system paths.
    void addSystemPaths();

    /// Control optional linker behavior by setting a group of flags. The flags
    /// are defined in the ControlFlags enumeration.
    /// @see ControlFlags
    /// @brief Set control flags.
    void setFlags(unsigned flags) { Flags = flags; }

    /// This method is the main interface to the linker. It can be used to
    /// link a set of linkage items into a module. A linkage item is either a
    /// file name with fully qualified path, or a library for which the Linker's
    /// LibraryPath will be utilized to locate the library. The bool value in
    /// the LinkItemKind should be set to true for libraries.  This function
    /// allows linking to preserve the order of specification associated with
    /// the command line, or for other purposes. Each item will be linked in
    /// turn as it occurs in \p Items.
    /// @returns true if an error occurred, false otherwise
    /// @see LinkItemKind
    /// @see getLastError
    /// @throws nothing
    bool LinkInItems (
      const ItemList& Items, ///< Set of libraries/files to link in
      ItemList& NativeItems  ///< Output list of native files/libs
    );

    /// This function links the bitcode \p Files into the composite module.
    /// Note that this does not do any linking of unresolved symbols. The \p
    /// Files are all completely linked into \p HeadModule regardless of
    /// unresolved symbols. This function just loads each bitcode file and
    /// calls LinkInModule on them.
    /// @returns true if an error occurs, false otherwise
    /// @see getLastError
    /// @brief Link in multiple files.
    bool LinkInFiles (
      const std::vector<sys::Path> & Files ///< Files to link in
    );

    /// This function links a single bitcode file, \p File, into the composite
    /// module. Note that this does not attempt to resolve symbols. This method
    /// just loads the bitcode file and calls LinkInModule on it. If an error
    /// occurs, the Linker's error string is set.
    /// @returns true if an error occurs, false otherwise
    /// @see getLastError
    /// @brief Link in a single file.
    bool LinkInFile(
      const sys::Path& File, ///< File to link in.
      bool &is_native        ///< Indicates if the file is native object file
    );

    /// This function provides a way to selectively link in a set of modules,
    /// found in libraries, based on the unresolved symbols in the composite
    /// module. Each item in \p Libraries should be the base name of a library,
    /// as if given with the -l option of a linker tool.  The Linker's LibPaths
    /// are searched for the \p Libraries and any found will be linked in with
    /// LinkInArchive.  If an error occurs, the Linker's error string is set.
    /// @see LinkInArchive
    /// @see getLastError
    /// @returns true if an error occurs, false otherwise
    /// @brief Link libraries into the module
    bool LinkInLibraries (
      const std::vector<std::string> & Libraries ///< Libraries to link in
    );

    /// This function provides a way to selectively link in a set of modules,
    /// found in one library, based on the unresolved symbols in the composite
    /// module.The \p Library should be the base name of a library, as if given
    /// with the -l option of a linker tool. The Linker's LibPaths are searched
    /// for the \p Library and if found, it will be linked in with via the
    /// LinkInArchive method. If an error occurs, the Linker's error string is
    /// set.
    /// @see LinkInArchive
    /// @see getLastError
    /// @returns true if an error occurs, false otherwise
    /// @brief Link one library into the module
    bool LinkInLibrary (
      const std::string& Library, ///< The library to link in
      bool& is_native             ///< Indicates if lib a native library
    );

    /// This function links one bitcode archive, \p Filename, into the module.
    /// The archive is searched to resolve outstanding symbols. Any modules in
    /// the archive that resolve outstanding symbols will be linked in. The
    /// library is searched repeatedly until no more modules that resolve
    /// symbols can be found. If an error occurs, the error string is  set.
    /// To speed up this function, ensure the the archive has been processed
    /// llvm-ranlib or the S option was given to llvm-ar when the archive was
    /// created. These tools add a symbol table to the archive which makes the
    /// search for undefined symbols much faster.
    /// @see getLastError
    /// @returns true if an error occurs, otherwise false.
    /// @brief Link in one archive.
    bool LinkInArchive(
      const sys::Path& Filename, ///< Filename of the archive to link
      bool& is_native            ///<  Indicates if archive is a native archive
    );

    /// This method links the \p Src module into the Linker's Composite module
    /// by calling LinkModules.  All the other LinkIn* methods eventually
    /// result in calling this method to link a Module into the Linker's
    /// composite.
    /// @see LinkModules
    /// @returns True if an error occurs, false otherwise.
    /// @brief Link in a module.
    bool LinkInModule(
      Module* Src,              ///< Module linked into \p Dest
      std::string* ErrorMsg = 0 /// Error/diagnostic string
    ) { 
      return LinkModules(Composite, Src, ErrorMsg ); 
    }

    /// This is the heart of the linker. This method will take unconditional
    /// control of the \p Src module and link it into the \p Dest module. The
    /// \p Src module will be destructed or subsumed by this method. In either
    /// case it is not usable by the caller after this method is invoked. Only
    /// the \p Dest module will remain. The \p Src module is linked into the
    /// Linker's composite module such that types, global variables, functions,
    /// and etc. are matched and resolved.  If an error occurs, this function
    /// returns true and ErrorMsg is set to a descriptive message about the
    /// error.
    /// @returns True if an error occurs, false otherwise.
    /// @brief Generically link two modules together.
    static bool LinkModules(Module* Dest, Module* Src, std::string* ErrorMsg);

    /// This function looks through the Linker's LibPaths to find a library with
    /// the name \p Filename. If the library cannot be found, the returned path
    /// will be empty (i.e. sys::Path::isEmpty() will return true).
    /// @returns A sys::Path to the found library
    /// @brief Find a library from its short name.
    sys::Path FindLib(const std::string &Filename);

  /// @}
  /// @name Implementation
  /// @{
  private:
    /// Read in and parse the bitcode file named by FN and return the
    /// Module it contains (wrapped in an auto_ptr), or 0 if an error occurs.
    std::auto_ptr<Module> LoadObject(const sys::Path& FN);

    bool warning(const std::string& message);
    bool error(const std::string& message);
    void verbose(const std::string& message);

  /// @}
  /// @name Data
  /// @{
  private:
    Module* Composite; ///< The composite module linked together
    std::vector<sys::Path> LibPaths; ///< The library search paths
    unsigned Flags;    ///< Flags to control optional behavior.
    std::string Error; ///< Text of error that occurred.
    std::string ProgramName; ///< Name of the program being linked
  /// @}

};

} // End llvm namespace

#endif
