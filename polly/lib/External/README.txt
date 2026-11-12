The libraries in this directory are mirrored from external projects.

Patches to them should first be contributed upstream and then return to Polly
as normal (re)imports of these updated libraries.

We currently have the following external libraries.

# isl
License: MIT-STYLE
Details: isl/LICENSE

# imath
License: MIT-STYLE
Details: isl/imath/LICENSE

To update these libraries run 'autoreconf -i && ./configure && make dist' in
the isl git directory and move the resulting files into lib/External/isl.
Alternatively, run the update-isl.sh script.
