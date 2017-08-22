#!/bin/sh

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script merges changes from the master copy of the Go library
# into the libgo library.  This does the easy stuff; the hard stuff is
# left to the user.

# The file MERGE should hold the Git revision number of the last
# revision which was merged into these sources.  Given that, and given
# the current sources, we can run the usual diff3 algorithm to merge
# all changes into our sources.

set -e

TMPDIR=${TMPDIR:-/tmp}

OLDDIR=${TMPDIR}/libgo-merge-old
NEWDIR=${TMPDIR}/libgo-merge-new

if ! test -f MERGE; then
  echo 1>&2 "merge.sh: must be run in libgo source directory"
  exit 1
fi

rev=weekly
case $# in
1) ;;
2) rev=$2 ;;
*)
  echo 1>&2 "merge.sh: Usage: merge.sh git-repository [revision]"
  exit 1
  ;;
esac

repository=$1

old_rev=`sed 1q MERGE`

rm -rf ${OLDDIR}
git clone ${repository} ${OLDDIR}
(cd ${OLDDIR} && git checkout ${old_rev})

rm -rf ${NEWDIR}
git clone ${repository} ${NEWDIR}
(cd ${NEWDIR} && git checkout ${rev})

new_rev=`cd ${NEWDIR} && git log | sed 1q | sed -e 's/commit //'`

merge() {
  name=$1
  old=$2
  new=$3
  libgo=$4
  if ! test -f ${new}; then
    # The file does not exist in the new version.
    if ! test -f ${old}; then
      echo 1>&2 "merge.sh internal error no files $old $new"
      exit 1
    fi
    if ! test -f ${libgo}; then
      # File removed in new version and libgo.
      :;
    else
      echo "merge.sh: ${name}: REMOVED"
      rm -f ${libgo}
      git rm ${libgo}
    fi
  elif test -f ${old}; then
    # The file exists in the old version.
    if ! test -f ${libgo}; then
      echo "merge.sh: $name: skipping: exists in old and new git, but not in libgo"
      continue
    fi
    if cmp -s ${old} ${libgo}; then
      # The libgo file is unchanged from the old version.
      if cmp -s ${new} ${libgo}; then
        # File is unchanged from old to new version.
        continue
      fi
      # Update file in libgo.
      echo "merge.sh: $name: updating"
      cp ${new} ${libgo}
    else
      # The libgo file has local changes.
      set +e
      diff3 -m -E ${libgo} ${old} ${new} > ${libgo}.tmp
      status=$?
      set -e
      case $status in
      0)
        echo "merge.sh: $name: updating"
        mv ${libgo}.tmp ${libgo}
        ;;
      1)
        echo "merge.sh: $name: CONFLICTS"
        mv ${libgo}.tmp ${libgo}
        ;;
      *)
        echo 1>&2 "merge.sh: $name: diff3 failure"
        exit 1
        ;;
      esac
    fi
  else
    # The file does not exist in the old version.
    if test -f ${libgo}; then
      if ! cmp -s ${new} ${libgo}; then
        echo 1>&2 "merge.sh: $name: IN NEW AND LIBGO BUT NOT OLD"
      fi
    else
      echo "merge.sh: $name: NEW"
      dir=`dirname ${libgo}`
      if ! test -d ${dir}; then
        mkdir -p ${dir}
      fi
      cp ${new} ${libgo}
      git add ${libgo}
    fi
  fi
}

merge_c() {
  from=$1
  to=$2
  oldfile=${OLDDIR}/src/runtime/$from
  if test -f ${oldfile}; then
    sed -e 's/·/_/g' < ${oldfile} > ${oldfile}.tmp
    oldfile=${oldfile}.tmp
    newfile=${NEWDIR}/src/runtime/$from
    sed -e 's/·/_/g' < ${newfile} > ${newfile}.tmp
    newfile=${newfile}.tmp
    libgofile=runtime/$to
    merge $from ${oldfile} ${newfile} ${libgofile}
  fi
}

if test -f VERSION; then
  if ! cmp -s ${NEWDIR}/VERSION VERSION; then
    cp ${NEWDIR}/VERSION .
  fi
else
  if test -f ${NEWDIR}/VERSION; then
    cp ${NEWDIR}/VERSION .
  fi
fi

(cd ${NEWDIR}/src && find . -name '*.go' -print) | while read f; do
  oldfile=${OLDDIR}/src/$f
  newfile=${NEWDIR}/src/$f
  libgofile=go/$f
  merge $f ${oldfile} ${newfile} ${libgofile}
done

(cd ${NEWDIR}/src && find . -name testdata -print) | while read d; do
  oldtd=${OLDDIR}/src/$d
  newtd=${NEWDIR}/src/$d
  libgotd=go/$d
  if ! test -d ${oldtd}; then
    continue
  fi
  (cd ${oldtd} && git ls-files .) | while read f; do
    if test "`basename $f`" = ".gitignore"; then
      continue
    fi
    name=$d/$f
    oldfile=${oldtd}/$f
    newfile=${newtd}/$f
    libgofile=${libgotd}/$f
    merge ${name} ${oldfile} ${newfile} ${libgofile}
  done
done

runtime="chan.goc chan.h cpuprof.goc env_posix.c heapdump.c lock_futex.c lfstack.goc lock_sema.c mcache.c mcentral.c mfixalloc.c mgc0.c mgc0.h mheap.c msize.c netpoll.goc netpoll_epoll.c netpoll_kqueue.c netpoll_stub.c panic.c print.c proc.c race.h rdebug.goc runtime.c runtime.h signal_unix.c signal_unix.h malloc.h malloc.goc mprof.goc parfor.c runtime1.goc sema.goc sigqueue.goc string.goc time.goc"
for f in $runtime; do
  # merge_c $f $f
  true
done

# merge_c os_linux.c thread-linux.c
# merge_c mem_linux.c mem.c

(cd ${OLDDIR}/src && find . -name '*.go' -print) | while read f; do
  oldfile=${OLDDIR}/src/$f
  newfile=${NEWDIR}/src/$f
  libgofile=go/$f
  if test -f ${newfile}; then
    continue
  fi
  if ! test -f ${libgofile}; then
    continue
  fi
  echo "merge.sh: ${libgofile}: REMOVED"
  rm -f ${libgofile}
  git rm ${libgofile}
done

(echo ${new_rev}; sed -ne '2,$p' MERGE) > MERGE.tmp
mv MERGE.tmp MERGE
