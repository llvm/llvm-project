//===-- TTYState.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 3/26/07.
//
//===----------------------------------------------------------------------===//

#ifndef __TTYState_h__
#define __TTYState_h__

#include <termios.h>
#include <stdint.h>

class TTYState
{
public:
    TTYState();
    ~TTYState();

    bool    GetTTYState (int fd, bool saveProcessGroup);
    bool    SetTTYState () const;

    bool    IsValid() const { return FileDescriptorValid() && TFlagsValid() && TTYStateValid(); }
    bool    FileDescriptorValid() const { return m_fd >= 0; }
    bool    TFlagsValid() const { return m_tflags != -1; }
    bool    TTYStateValid() const { return m_ttystateErr == 0; }
    bool    ProcessGroupValid() const { return m_processGroup != -1; }

protected:
    int             m_fd;                // File descriptor
    int             m_tflags;
    int             m_ttystateErr;
    struct termios  m_ttystate;
    pid_t           m_processGroup;

};


class TTYStateSwitcher
{
public:
    TTYStateSwitcher();
    ~TTYStateSwitcher();

    bool GetState(uint32_t idx, int fd, bool saveProcessGroup);
    bool SetState(uint32_t idx) const;
    uint32_t NumStates() const { return sizeof(m_ttystates)/sizeof(TTYState); }
    bool ValidStateIndex(uint32_t idx) const { return idx < NumStates(); }

protected:
    mutable uint32_t    m_currentState;
    TTYState            m_ttystates[2];
};

#endif