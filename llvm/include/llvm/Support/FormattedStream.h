//===-- llvm/Support/FormattedStream.h - Formatted streams ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to do
// things like pretty-print comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMATTEDSTREAM_H
#define LLVM_SUPPORT_FORMATTEDSTREAM_H

#include "llvm/Support/raw_ostream.h"

namespace llvm 
{
  /// formatted_raw_ostream - Formatted raw_fd_ostream to handle
  /// asm-specific constructs.
  ///
  class formatted_raw_ostream : public raw_ostream {
  public:
    /// DELETE_STREAM - Tell the destructor to delete the held stream.
    ///
    static const bool DELETE_STREAM = true;

    /// PRESERVE_STREAM - Tell the destructor to not delete the held
    /// stream.
    ///
    static const bool PRESERVE_STREAM = false;

  private:
    /// TheStream - The real stream we output to. We set it to be
    /// unbuffered, since we're already doing our own buffering.
    ///
    raw_ostream *TheStream;

    /// DeleteStream - Do we need to delete TheStream in the
    /// destructor?
    ///
    bool DeleteStream;

    /// ColumnScanned - The current output column of the data that's
    /// been flushed and the portion of the buffer that's been
    /// scanned.  The column scheme is zero-based.
    ///
    unsigned ColumnScanned;

    /// Scanned - This points to one past the last character in the
    /// buffer we've scanned.
    ///
    const char *Scanned;

    virtual void write_impl(const char *Ptr, size_t Size) LLVM_OVERRIDE;

    /// current_pos - Return the current position within the stream,
    /// not counting the bytes currently in the buffer.
    virtual uint64_t current_pos() const LLVM_OVERRIDE {
      // Our current position in the stream is all the contents which have been
      // written to the underlying stream (*not* the current position of the
      // underlying stream).
      return TheStream->tell();
    }

    /// ComputeColumn - Examine the given output buffer and figure out which
    /// column we end up in after output.
    ///
    void ComputeColumn(const char *Ptr, size_t size);

  public:
    /// formatted_raw_ostream - Open the specified file for
    /// writing. If an error occurs, information about the error is
    /// put into ErrorInfo, and the stream should be immediately
    /// destroyed; the string will be empty if no error occurred.
    ///
    /// As a side effect, the given Stream is set to be Unbuffered.
    /// This is because formatted_raw_ostream does its own buffering,
    /// so it doesn't want another layer of buffering to be happening
    /// underneath it.
    ///
    formatted_raw_ostream(raw_ostream &Stream, bool Delete = false) 
      : raw_ostream(), TheStream(0), DeleteStream(false), ColumnScanned(0) {
      setStream(Stream, Delete);
    }
    explicit formatted_raw_ostream()
      : raw_ostream(), TheStream(0), DeleteStream(false), ColumnScanned(0) {
      Scanned = 0;
    }

    ~formatted_raw_ostream() {
      flush();
      releaseStream();
    }

    void setStream(raw_ostream &Stream, bool Delete = false) {
      releaseStream();

      TheStream = &Stream;
      DeleteStream = Delete;

      // This formatted_raw_ostream inherits from raw_ostream, so it'll do its
      // own buffering, and it doesn't need or want TheStream to do another
      // layer of buffering underneath. Resize the buffer to what TheStream
      // had been using, and tell TheStream not to do its own buffering.
      if (size_t BufferSize = TheStream->GetBufferSize())
        SetBufferSize(BufferSize);
      else
        SetUnbuffered();
      TheStream->SetUnbuffered();

      Scanned = 0;
    }

    /// PadToColumn - Align the output to some column number.  If the current
    /// column is already equal to or more than NewCol, PadToColumn inserts one
    /// space.
    ///
    /// \param NewCol - The column to move to.
    formatted_raw_ostream &PadToColumn(unsigned NewCol);

  private:
    void releaseStream() {
      // Delete the stream if needed. Otherwise, transfer the buffer
      // settings from this raw_ostream back to the underlying stream.
      if (!TheStream)
        return;
      if (DeleteStream)
        delete TheStream;
      else if (size_t BufferSize = GetBufferSize())
        TheStream->SetBufferSize(BufferSize);
      else
        TheStream->SetUnbuffered();
    }
  };

/// fouts() - This returns a reference to a formatted_raw_ostream for
/// standard output.  Use it like: fouts() << "foo" << "bar";
formatted_raw_ostream &fouts();

/// ferrs() - This returns a reference to a formatted_raw_ostream for
/// standard error.  Use it like: ferrs() << "foo" << "bar";
formatted_raw_ostream &ferrs();

/// fdbgs() - This returns a reference to a formatted_raw_ostream for
/// debug output.  Use it like: fdbgs() << "foo" << "bar";
formatted_raw_ostream &fdbgs();

} // end llvm namespace


#endif
