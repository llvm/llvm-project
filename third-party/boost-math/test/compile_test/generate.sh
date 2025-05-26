for file in ../../../../boost/math/tools/*.hpp; do 
cat > tools_$(basename $file .hpp)_inc_test.cpp  << EOF; 
//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tools/$(basename $file)>
// #includes all the files that it needs to.
//
#include <boost/math/tools/$(basename $file .hpp).hpp>

EOF
done

for file in ../../../../boost/math/distributions/*.hpp; do 
cat > dist_$(basename $file .hpp)_incl_test.cpp  << EOF; 
//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/distributions/$(basename $file)>
// #includes all the files that it needs to.
//
#include <boost/math/distributions/$(basename $file .hpp).hpp>

EOF
done

for file in ../../../../boost/math/special_functions/*.hpp; do 
cat > sf_$(basename $file .hpp)_incl_test.cpp  << EOF; 
//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/$(basename $file)>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/$(basename $file .hpp).hpp>

EOF
done


