//===- VettedPeer.h - A channel whose peer has been vetted ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A channel, marked with the basis on which this process chose to trust the
// peer at its other end.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_VETTEDPEER_H
#define ORC_RT_BEDROCK_VETTEDPEER_H

#include <utility>

namespace orc_rt {

/// A channel to a peer that this process has decided to trust.
///
/// The peer at the other end of a controller channel can send this process code
/// to run, so deciding to trust it is the most security-sensitive decision a
/// JIT executor makes. Interfaces that start a conversation over a channel take
/// a VettedPeer rather than a bare channel, so the decision can't be skipped
/// by omission: the only way to make a VettedPeer is through one of the
/// factories below, and each one names a reason for trusting the peer.
///
/// A VettedPeer can't check that the reason is sound, only that one was chosen.
/// Because each reason is a named factory, every choice shows up in the source
/// and is easy to search for (e.g. "VettedPeer<SocketHandle>::unchecked").
template <typename ChannelT> class VettedPeer {
public:
  /// The channel was handed to this process by whoever started it, e.g. a
  /// socket descriptor inherited across exec.
  ///
  /// Trusting this peer grants it nothing new, since whoever started this
  /// process could already have made it run any code they liked. That holds
  /// only if starting it gained no privilege, which the caller must ensure: no
  /// setuid or setgid, no file capabilities, no change of security domain (e.g.
  /// SELinux or AppArmor), and no added entitlements.
  static VettedPeer inherited(ChannelT C) { return VettedPeer(std::move(C)); }

  /// The peer's identity was checked against a requirement before this was
  /// made, e.g. a unix domain socket peer's user id, or an XPC peer's code
  /// signature.
  static VettedPeer checked(ChannelT C) { return VettedPeer(std::move(C)); }

  /// The peer is not identified, e.g. the peer of a TCP connection, which is
  /// known only by its address.
  ///
  /// Anyone who can impersonate the peer, or anyone at all who can reach a
  /// listening endpoint, can run code in this process.
  static VettedPeer unchecked(ChannelT C) { return VettedPeer(std::move(C)); }

  /// Gives up the channel, for the interface that this VettedPeer was made for.
  ChannelT take() && { return std::move(C); }

private:
  explicit VettedPeer(ChannelT C) : C(std::move(C)) {}

  ChannelT C;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_VETTEDPEER_H
