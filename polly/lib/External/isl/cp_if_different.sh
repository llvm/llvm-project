#/bin/sh
# Copy $SRC to $DST unless $DST already exists and has the same contents.
SRC="$1"
DST="$2"

if cmp -s "$SRC" "$DST" 2>/dev/null; then
    :
else
    cp "$SRC" "$DST"
fi
