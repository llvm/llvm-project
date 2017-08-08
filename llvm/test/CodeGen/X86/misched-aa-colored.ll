; RUN: llc < %s -mcpu=x86-64 -enable-misched -misched-bottomup=0 -misched-topdown=0 -misched=shuffle -enable-aa-sched-mi | FileCheck %s
; REQUIRES: asserts
; -misched=shuffle is NDEBUG only!

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.llvm::SDNode.10.610.970.1930.2050.2290.4090" = type { %"class.llvm::FoldingSetImpl::Node.0.600.960.1920.2040.2280.4080", %"class.llvm::ilist_node.2.602.962.1922.2042.2282.4082", i16, [2 x i8], i32, %"class.llvm::SDUse.4.604.964.1924.2044.2284.4084"*, %"struct.llvm::EVT.8.608.968.1928.2048.2288.4088"*, %"class.llvm::SDUse.4.604.964.1924.2044.2284.4084"*, i16, i16, %"class.llvm::DebugLoc.9.609.969.1929.2049.2289.4089", i32 }
%"class.llvm::FoldingSetImpl::Node.0.600.960.1920.2040.2280.4080" = type { i8* }
%"class.llvm::ilist_node.2.602.962.1922.2042.2282.4082" = type { %"class.llvm::ilist_half_node.1.601.961.1921.2041.2281.4081", %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"* }
%"class.llvm::ilist_half_node.1.601.961.1921.2041.2281.4081" = type { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"* }
%"struct.llvm::EVT.8.608.968.1928.2048.2288.4088" = type { %"class.llvm::MVT.5.605.965.1925.2045.2285.4085", %"class.llvm::Type.7.607.967.1927.2047.2287.4087"* }
%"class.llvm::MVT.5.605.965.1925.2045.2285.4085" = type { i32 }
%"class.llvm::SDUse.4.604.964.1924.2044.2284.4084" = type { %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083", %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, %"class.llvm::SDUse.4.604.964.1924.2044.2284.4084"**, %"class.llvm::SDUse.4.604.964.1924.2044.2284.4084"* }
%"class.llvm::SDValue.3.603.963.1923.2043.2283.4083" = type { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, i32 }
%"class.llvm::DebugLoc.9.609.969.1929.2049.2289.4089" = type { i32, i32 }
%"class.llvm::SelectionDAG.104.704.1064.2024.2144.2384.4184" = type { %"class.llvm::TargetMachine.17.617.977.1937.2057.2297.4097"*, %"class.llvm::TargetSelectionDAGInfo.18.618.978.1938.2058.2298.4098"*, %"class.llvm::TargetTransformInfo.19.619.979.1939.2059.2299.4099"*, %"class.llvm::TargetLowering.51.651.1011.1971.2091.2331.4131"*, %"class.llvm::MachineFunction.52.652.1012.1972.2092.2332.4132"*, %"class.llvm::LLVMContext.6.606.966.1926.2046.2286.4086"*, i32, %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090", %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083", %"struct.llvm::ilist.55.655.1015.1975.2095.2335.4135", %"class.llvm::RecyclingAllocator.65.665.1025.1985.2105.2345.4145", %"class.llvm::FoldingSet.67.667.1027.1987.2107.2347.4147", %"class.llvm::BumpPtrAllocator.64.664.1024.1984.2104.2344.4144", %"class.llvm::BumpPtrAllocator.64.664.1024.1984.2104.2344.4144", %"class.llvm::SDDbgInfo.79.679.1039.1999.2119.2359.4159"*, i8, %"struct.llvm::SelectionDAG::DAGUpdateListener.80.680.1040.2000.2120.2360.4160"*, %"class.std::map.43.84.684.1044.2004.2124.2364.4164", %"class.llvm::FoldingSet.50.85.685.1045.2005.2125.2365.4165", %"class.std::vector.51.89.689.1049.2009.2129.2369.4169", %"class.std::vector.56.92.692.1052.2012.2132.2372.4172", %"class.std::map.61.96.696.1056.2016.2136.2376.4176", %"class.llvm::StringMap.99.699.1059.2019.2139.2379.4179", %"class.std::map.66.103.703.1063.2023.2143.2383.4183" }
%"class.llvm::TargetMachine.17.617.977.1937.2057.2297.4097" = type { i32 (...)**, %"class.llvm::Target.11.611.971.1931.2051.2291.4091"*, %"class.std::basic_string.13.613.973.1933.2053.2293.4093", %"class.std::basic_string.13.613.973.1933.2053.2293.4093", %"class.std::basic_string.13.613.973.1933.2053.2293.4093", %"class.llvm::MCCodeGenInfo.14.614.974.1934.2054.2294.4094"*, %"class.llvm::MCAsmInfo.15.615.975.1935.2055.2295.4095"*, i8, %"class.llvm::TargetOptions.16.616.976.1936.2056.2296.4096" }
%"class.llvm::Target.11.611.971.1931.2051.2291.4091" = type opaque
%"class.std::basic_string.13.613.973.1933.2053.2293.4093" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.12.612.972.1932.2052.2292.4092" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.12.612.972.1932.2052.2292.4092" = type { i8* }
%"class.llvm::MCCodeGenInfo.14.614.974.1934.2054.2294.4094" = type opaque
%"class.llvm::MCAsmInfo.15.615.975.1935.2055.2295.4095" = type opaque
%"class.llvm::TargetOptions.16.616.976.1936.2056.2296.4096" = type { [2 x i8], i32, i8, %"class.std::basic_string.13.613.973.1933.2053.2293.4093", i32, i32 }
%"class.llvm::TargetSelectionDAGInfo.18.618.978.1938.2058.2298.4098" = type opaque
%"class.llvm::TargetTransformInfo.19.619.979.1939.2059.2299.4099" = type opaque
%"class.llvm::TargetLowering.51.651.1011.1971.2091.2331.4131" = type { %"class.llvm::TargetLoweringBase.50.650.1010.1970.2090.2330.4130" }
%"class.llvm::TargetLoweringBase.50.650.1010.1970.2090.2330.4130" = type { i32 (...)**, %"class.llvm::TargetMachine.17.617.977.1937.2057.2297.4097"*, %"class.llvm::DataLayout.35.635.995.1955.2075.2315.4115"*, %"class.llvm::TargetLoweringObjectFile.36.636.996.1956.2076.2316.4116"*, i8, i8, i8, i8, %"class.llvm::DenseMap.11.38.638.998.1958.2078.2318.4118", i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, [58 x %"class.llvm::TargetRegisterClass.39.639.999.1959.2079.2319.4119"*], [58 x i8], [58 x %"class.llvm::MVT.5.605.965.1925.2045.2285.4085"], [58 x %"class.llvm::TargetRegisterClass.39.639.999.1959.2079.2319.4119"*], [58 x i8], [58 x %"class.llvm::MVT.5.605.965.1925.2045.2285.4085"], [58 x [188 x i8]], [58 x [4 x i8]], [58 x [58 x i8]], [58 x [5 x i8]], [24 x [4 x i32]], %"class.llvm::TargetLoweringBase::ValueTypeActionImpl.40.640.1000.1960.2080.2320.4120", %"class.std::vector.15.44.644.1004.1964.2084.2324.4124", [24 x i8], %"class.std::map.49.649.1009.1969.2089.2329.4129", [341 x i8*], [341 x i32], [341 x i32], i32, i32, i32, i32, i32, i32, i8 }
%"class.llvm::DataLayout.35.635.995.1955.2075.2315.4115" = type { [28 x i8], i8, i32, i32, %"class.llvm::SmallVector.23.623.983.1943.2063.2303.4103", %"class.llvm::SmallVector.3.31.631.991.1951.2071.2311.4111", %"class.llvm::DenseMap.34.634.994.1954.2074.2314.4114", i8* }
%"class.llvm::SmallVector.23.623.983.1943.2063.2303.4103" = type { [25 x i8], %"struct.llvm::SmallVectorStorage.22.622.982.1942.2062.2302.4102" }
%"struct.llvm::SmallVectorStorage.22.622.982.1942.2062.2302.4102" = type { [7 x %"struct.llvm::AlignedCharArrayUnion.21.621.981.1941.2061.2301.4101"] }
%"struct.llvm::AlignedCharArrayUnion.21.621.981.1941.2061.2301.4101" = type { %"struct.llvm::AlignedCharArray.20.620.980.1940.2060.2300.4100" }
%"struct.llvm::AlignedCharArray.20.620.980.1940.2060.2300.4100" = type { [1 x i8] }
%"class.llvm::SmallVector.3.31.631.991.1951.2071.2311.4111" = type { %"class.llvm::SmallVectorImpl.4.29.629.989.1949.2069.2309.4109", %"struct.llvm::SmallVectorStorage.9.30.630.990.1950.2070.2310.4110" }
%"class.llvm::SmallVectorImpl.4.29.629.989.1949.2069.2309.4109" = type { %"class.llvm::SmallVectorTemplateBase.5.28.628.988.1948.2068.2308.4108" }
%"class.llvm::SmallVectorTemplateBase.5.28.628.988.1948.2068.2308.4108" = type { %"class.llvm::SmallVectorTemplateCommon.6.27.627.987.1947.2067.2307.4107" }
%"class.llvm::SmallVectorTemplateCommon.6.27.627.987.1947.2067.2307.4107" = type { %"class.llvm::SmallVectorBase.24.624.984.1944.2064.2304.4104", %"struct.llvm::AlignedCharArrayUnion.7.26.626.986.1946.2066.2306.4106" }
%"class.llvm::SmallVectorBase.24.624.984.1944.2064.2304.4104" = type { i8*, i8*, i8* }
%"struct.llvm::AlignedCharArrayUnion.7.26.626.986.1946.2066.2306.4106" = type { %"struct.llvm::AlignedCharArray.8.25.625.985.1945.2065.2305.4105" }
%"struct.llvm::AlignedCharArray.8.25.625.985.1945.2065.2305.4105" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage.9.30.630.990.1950.2070.2310.4110" = type { [15 x %"struct.llvm::AlignedCharArrayUnion.7.26.626.986.1946.2066.2306.4106"] }
%"class.llvm::DenseMap.34.634.994.1954.2074.2314.4114" = type { %"struct.std::pair.10.33.633.993.1953.2073.2313.4113"*, i32, i32, i32 }
%"struct.std::pair.10.33.633.993.1953.2073.2313.4113" = type { i32, %"struct.llvm::PointerAlignElem.32.632.992.1952.2072.2312.4112" }
%"struct.llvm::PointerAlignElem.32.632.992.1952.2072.2312.4112" = type { i32, i32, i32, i32 }
%"class.llvm::TargetLoweringObjectFile.36.636.996.1956.2076.2316.4116" = type opaque
%"class.llvm::DenseMap.11.38.638.998.1958.2078.2318.4118" = type { %"struct.std::pair.14.37.637.997.1957.2077.2317.4117"*, i32, i32, i32 }
%"struct.std::pair.14.37.637.997.1957.2077.2317.4117" = type { i32, i32 }
%"class.llvm::TargetRegisterClass.39.639.999.1959.2079.2319.4119" = type opaque
%"class.llvm::TargetLoweringBase::ValueTypeActionImpl.40.640.1000.1960.2080.2320.4120" = type { [58 x i8] }
%"class.std::vector.15.44.644.1004.1964.2084.2324.4124" = type { %"struct.std::_Vector_base.16.43.643.1003.1963.2083.2323.4123" }
%"struct.std::_Vector_base.16.43.643.1003.1963.2083.2323.4123" = type { %"struct.std::_Vector_base<std::pair<llvm::MVT, const llvm::TargetRegisterClass *>, std::allocator<std::pair<llvm::MVT, const llvm::TargetRegisterClass *> > >::_Vector_impl.42.642.1002.1962.2082.2322.4122" }
%"struct.std::_Vector_base<std::pair<llvm::MVT, const llvm::TargetRegisterClass *>, std::allocator<std::pair<llvm::MVT, const llvm::TargetRegisterClass *> > >::_Vector_impl.42.642.1002.1962.2082.2322.4122" = type { %"struct.std::pair.20.41.641.1001.1961.2081.2321.4121"*, %"struct.std::pair.20.41.641.1001.1961.2081.2321.4121"*, %"struct.std::pair.20.41.641.1001.1961.2081.2321.4121"* }
%"struct.std::pair.20.41.641.1001.1961.2081.2321.4121" = type { %"class.llvm::MVT.5.605.965.1925.2045.2285.4085", %"class.llvm::TargetRegisterClass.39.639.999.1959.2079.2319.4119"* }
%"class.std::map.49.649.1009.1969.2089.2329.4129" = type { %"class.std::_Rb_tree.48.648.1008.1968.2088.2328.4128" }
%"class.std::_Rb_tree.48.648.1008.1968.2088.2328.4128" = type { %"struct.std::_Rb_tree<std::pair<unsigned int, llvm::MVT::SimpleValueType>, std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType>, std::_Select1st<std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType> >, std::less<std::pair<unsigned int, llvm::MVT::SimpleValueType> >, std::allocator<std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType> > >::_Rb_tree_impl.47.647.1007.1967.2087.2327.4127" }
%"struct.std::_Rb_tree<std::pair<unsigned int, llvm::MVT::SimpleValueType>, std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType>, std::_Select1st<std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType> >, std::less<std::pair<unsigned int, llvm::MVT::SimpleValueType> >, std::allocator<std::pair<const std::pair<unsigned int, llvm::MVT::SimpleValueType>, llvm::MVT::SimpleValueType> > >::_Rb_tree_impl.47.647.1007.1967.2087.2327.4127" = type { %"struct.std::less.45.645.1005.1965.2085.2325.4125", %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126", i64 }
%"struct.std::less.45.645.1005.1965.2085.2325.4125" = type { i8 }
%"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126" = type { i32, %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126"*, %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126"*, %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126"* }
%"class.llvm::MachineFunction.52.652.1012.1972.2092.2332.4132" = type opaque
%"class.llvm::LLVMContext.6.606.966.1926.2046.2286.4086" = type opaque
%"struct.llvm::ilist.55.655.1015.1975.2095.2335.4135" = type { %"class.llvm::iplist.54.654.1014.1974.2094.2334.4134" }
%"class.llvm::iplist.54.654.1014.1974.2094.2334.4134" = type { %"struct.llvm::ilist_traits.53.653.1013.1973.2093.2333.4133", %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"* }
%"struct.llvm::ilist_traits.53.653.1013.1973.2093.2333.4133" = type { %"class.llvm::ilist_half_node.1.601.961.1921.2041.2281.4081" }
%"class.llvm::RecyclingAllocator.65.665.1025.1985.2105.2345.4145" = type { %"class.llvm::Recycler.59.659.1019.1979.2099.2339.4139", %"class.llvm::BumpPtrAllocator.64.664.1024.1984.2104.2344.4144" }
%"class.llvm::Recycler.59.659.1019.1979.2099.2339.4139" = type { %"class.llvm::iplist.24.58.658.1018.1978.2098.2338.4138" }
%"class.llvm::iplist.24.58.658.1018.1978.2098.2338.4138" = type { %"struct.llvm::ilist_traits.25.57.657.1017.1977.2097.2337.4137", %"struct.llvm::RecyclerStruct.56.656.1016.1976.2096.2336.4136"* }
%"struct.llvm::ilist_traits.25.57.657.1017.1977.2097.2337.4137" = type { %"struct.llvm::RecyclerStruct.56.656.1016.1976.2096.2336.4136" }
%"struct.llvm::RecyclerStruct.56.656.1016.1976.2096.2336.4136" = type { %"struct.llvm::RecyclerStruct.56.656.1016.1976.2096.2336.4136"*, %"struct.llvm::RecyclerStruct.56.656.1016.1976.2096.2336.4136"* }
%"class.llvm::FoldingSet.67.667.1027.1987.2107.2347.4147" = type { %"class.llvm::FoldingSetImpl.66.666.1026.1986.2106.2346.4146" }
%"class.llvm::FoldingSetImpl.66.666.1026.1986.2106.2346.4146" = type { i32 (...)**, i8**, i32, i32 }
%"class.llvm::BumpPtrAllocator.64.664.1024.1984.2104.2344.4144" = type { i64, i64, %"class.llvm::MallocSlabAllocator.62.662.1022.1982.2102.2342.4142", %"class.llvm::SlabAllocator.60.660.1020.1980.2100.2340.4140"*, %"class.llvm::MemSlab.63.663.1023.1983.2103.2343.4143"*, i8*, i8*, i64 }
%"class.llvm::MallocSlabAllocator.62.662.1022.1982.2102.2342.4142" = type { %"class.llvm::SlabAllocator.60.660.1020.1980.2100.2340.4140", %"class.llvm::MallocAllocator.61.661.1021.1981.2101.2341.4141" }
%"class.llvm::SlabAllocator.60.660.1020.1980.2100.2340.4140" = type { i32 (...)** }
%"class.llvm::MallocAllocator.61.661.1021.1981.2101.2341.4141" = type { i8 }
%"class.llvm::MemSlab.63.663.1023.1983.2103.2343.4143" = type { i64, %"class.llvm::MemSlab.63.663.1023.1983.2103.2343.4143"* }
%"class.llvm::SDDbgInfo.79.679.1039.1999.2119.2359.4159" = type { %"class.llvm::SmallVector.30.74.674.1034.1994.2114.2354.4154", %"class.llvm::SmallVector.30.74.674.1034.1994.2114.2354.4154", %"class.llvm::DenseMap.37.78.678.1038.1998.2118.2358.4158" }
%"class.llvm::SmallVector.30.74.674.1034.1994.2114.2354.4154" = type { %"class.llvm::SmallVectorImpl.31.72.672.1032.1992.2112.2352.4152", %"struct.llvm::SmallVectorStorage.36.73.673.1033.1993.2113.2353.4153" }
%"class.llvm::SmallVectorImpl.31.72.672.1032.1992.2112.2352.4152" = type { %"class.llvm::SmallVectorTemplateBase.32.71.671.1031.1991.2111.2351.4151" }
%"class.llvm::SmallVectorTemplateBase.32.71.671.1031.1991.2111.2351.4151" = type { %"class.llvm::SmallVectorTemplateCommon.33.70.670.1030.1990.2110.2350.4150" }
%"class.llvm::SmallVectorTemplateCommon.33.70.670.1030.1990.2110.2350.4150" = type { %"class.llvm::SmallVectorBase.24.624.984.1944.2064.2304.4104", %"struct.llvm::AlignedCharArrayUnion.34.69.669.1029.1989.2109.2349.4149" }
%"struct.llvm::AlignedCharArrayUnion.34.69.669.1029.1989.2109.2349.4149" = type { %"struct.llvm::AlignedCharArray.35.68.668.1028.1988.2108.2348.4148" }
%"struct.llvm::AlignedCharArray.35.68.668.1028.1988.2108.2348.4148" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage.36.73.673.1033.1993.2113.2353.4153" = type { [31 x %"struct.llvm::AlignedCharArrayUnion.34.69.669.1029.1989.2109.2349.4149"] }
%"class.llvm::DenseMap.37.78.678.1038.1998.2118.2358.4158" = type { %"struct.std::pair.40.77.677.1037.1997.2117.2357.4157"*, i32, i32, i32 }
%"struct.std::pair.40.77.677.1037.1997.2117.2357.4157" = type { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, %"class.llvm::SmallVector.41.76.676.1036.1996.2116.2356.4156" }
%"class.llvm::SmallVector.41.76.676.1036.1996.2116.2356.4156" = type { %"class.llvm::SmallVectorImpl.31.72.672.1032.1992.2112.2352.4152", %"struct.llvm::SmallVectorStorage.42.75.675.1035.1995.2115.2355.4155" }
%"struct.llvm::SmallVectorStorage.42.75.675.1035.1995.2115.2355.4155" = type { [1 x %"struct.llvm::AlignedCharArrayUnion.34.69.669.1029.1989.2109.2349.4149"] }
%"struct.llvm::SelectionDAG::DAGUpdateListener.80.680.1040.2000.2120.2360.4160" = type { i32 (...)**, %"struct.llvm::SelectionDAG::DAGUpdateListener.80.680.1040.2000.2120.2360.4160"*, %"class.llvm::SelectionDAG.104.704.1064.2024.2144.2384.4184"* }
%"class.std::map.43.84.684.1044.2004.2124.2364.4164" = type { %"class.std::_Rb_tree.44.83.683.1043.2003.2123.2363.4163" }
%"class.std::_Rb_tree.44.83.683.1043.2003.2123.2363.4163" = type { %"struct.std::_Rb_tree<const llvm::SDNode *, std::pair<const llvm::SDNode *const, std::basic_string<char> >, std::_Select1st<std::pair<const llvm::SDNode *const, std::basic_string<char> > >, std::less<const llvm::SDNode *>, std::allocator<std::pair<const llvm::SDNode *const, std::basic_string<char> > > >::_Rb_tree_impl.82.682.1042.2002.2122.2362.4162" }
%"struct.std::_Rb_tree<const llvm::SDNode *, std::pair<const llvm::SDNode *const, std::basic_string<char> >, std::_Select1st<std::pair<const llvm::SDNode *const, std::basic_string<char> > >, std::less<const llvm::SDNode *>, std::allocator<std::pair<const llvm::SDNode *const, std::basic_string<char> > > >::_Rb_tree_impl.82.682.1042.2002.2122.2362.4162" = type { %"struct.std::less.48.81.681.1041.2001.2121.2361.4161", %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126", i64 }
%"struct.std::less.48.81.681.1041.2001.2121.2361.4161" = type { i8 }
%"class.llvm::FoldingSet.50.85.685.1045.2005.2125.2365.4165" = type { %"class.llvm::FoldingSetImpl.66.666.1026.1986.2106.2346.4146" }
%"class.std::vector.51.89.689.1049.2009.2129.2369.4169" = type { %"struct.std::_Vector_base.52.88.688.1048.2008.2128.2368.4168" }
%"struct.std::_Vector_base.52.88.688.1048.2008.2128.2368.4168" = type { %"struct.std::_Vector_base<llvm::CondCodeSDNode *, std::allocator<llvm::CondCodeSDNode *> >::_Vector_impl.87.687.1047.2007.2127.2367.4167" }
%"struct.std::_Vector_base<llvm::CondCodeSDNode *, std::allocator<llvm::CondCodeSDNode *> >::_Vector_impl.87.687.1047.2007.2127.2367.4167" = type { %"class.llvm::CondCodeSDNode.86.686.1046.2006.2126.2366.4166"**, %"class.llvm::CondCodeSDNode.86.686.1046.2006.2126.2366.4166"**, %"class.llvm::CondCodeSDNode.86.686.1046.2006.2126.2366.4166"** }
%"class.llvm::CondCodeSDNode.86.686.1046.2006.2126.2366.4166" = type { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090", i32 }
%"class.std::vector.56.92.692.1052.2012.2132.2372.4172" = type { %"struct.std::_Vector_base.57.91.691.1051.2011.2131.2371.4171" }
%"struct.std::_Vector_base.57.91.691.1051.2011.2131.2371.4171" = type { %"struct.std::_Vector_base<llvm::SDNode *, std::allocator<llvm::SDNode *> >::_Vector_impl.90.690.1050.2010.2130.2370.4170" }
%"struct.std::_Vector_base<llvm::SDNode *, std::allocator<llvm::SDNode *> >::_Vector_impl.90.690.1050.2010.2130.2370.4170" = type { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"**, %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"**, %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"** }
%"class.std::map.61.96.696.1056.2016.2136.2376.4176" = type { %"class.std::_Rb_tree.62.95.695.1055.2015.2135.2375.4175" }
%"class.std::_Rb_tree.62.95.695.1055.2015.2135.2375.4175" = type { %"struct.std::_Rb_tree<llvm::EVT, std::pair<const llvm::EVT, llvm::SDNode *>, std::_Select1st<std::pair<const llvm::EVT, llvm::SDNode *> >, llvm::EVT::compareRawBits, std::allocator<std::pair<const llvm::EVT, llvm::SDNode *> > >::_Rb_tree_impl.94.694.1054.2014.2134.2374.4174" }
%"struct.std::_Rb_tree<llvm::EVT, std::pair<const llvm::EVT, llvm::SDNode *>, std::_Select1st<std::pair<const llvm::EVT, llvm::SDNode *> >, llvm::EVT::compareRawBits, std::allocator<std::pair<const llvm::EVT, llvm::SDNode *> > >::_Rb_tree_impl.94.694.1054.2014.2134.2374.4174" = type { %"struct.llvm::EVT::compareRawBits.93.693.1053.2013.2133.2373.4173", %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126", i64 }
%"struct.llvm::EVT::compareRawBits.93.693.1053.2013.2133.2373.4173" = type { i8 }
%"class.llvm::StringMap.99.699.1059.2019.2139.2379.4179" = type { %"class.llvm::StringMapImpl.98.698.1058.2018.2138.2378.4178", %"class.llvm::MallocAllocator.61.661.1021.1981.2101.2341.4141" }
%"class.llvm::StringMapImpl.98.698.1058.2018.2138.2378.4178" = type { %"class.llvm::StringMapEntryBase.97.697.1057.2017.2137.2377.4177"**, i32, i32, i32, i32 }
%"class.llvm::StringMapEntryBase.97.697.1057.2017.2137.2377.4177" = type { i32 }
%"class.std::map.66.103.703.1063.2023.2143.2383.4183" = type { %"class.std::_Rb_tree.67.102.702.1062.2022.2142.2382.4182" }
%"class.std::_Rb_tree.67.102.702.1062.2022.2142.2382.4182" = type { %"struct.std::_Rb_tree<std::pair<std::basic_string<char>, unsigned char>, std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *>, std::_Select1st<std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *> >, std::less<std::pair<std::basic_string<char>, unsigned char> >, std::allocator<std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *> > >::_Rb_tree_impl.101.701.1061.2021.2141.2381.4181" }
%"struct.std::_Rb_tree<std::pair<std::basic_string<char>, unsigned char>, std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *>, std::_Select1st<std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *> >, std::less<std::pair<std::basic_string<char>, unsigned char> >, std::allocator<std::pair<const std::pair<std::basic_string<char>, unsigned char>, llvm::SDNode *> > >::_Rb_tree_impl.101.701.1061.2021.2141.2381.4181" = type { %"struct.std::less.71.100.700.1060.2020.2140.2380.4180", %"struct.std::_Rb_tree_node_base.46.646.1006.1966.2086.2326.4126", i64 }
%"struct.std::less.71.100.700.1060.2020.2140.2380.4180" = type { i8 }
%"class.llvm::Type.7.607.967.1927.2047.2287.4087" = type { %"class.llvm::LLVMContext.6.606.966.1926.2046.2286.4086"*, i32, i32, %"class.llvm::Type.7.607.967.1927.2047.2287.4087"** }
%"class.llvm::DAGTypeLegalizer.117.717.1077.2037.2157.2397.4197" = type { %"class.llvm::TargetLowering.51.651.1011.1971.2091.2331.4131"*, %"class.llvm::SelectionDAG.104.704.1064.2024.2144.2384.4184"*, %"class.llvm::TargetLoweringBase::ValueTypeActionImpl.40.640.1000.1960.2080.2320.4120", [6 x i8], %"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187", %"class.llvm::SmallDenseMap.77.110.710.1070.2030.2150.2390.4190", %"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187", %"class.llvm::SmallDenseMap.77.110.710.1070.2030.2150.2390.4190", %"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187", %"class.llvm::SmallDenseMap.77.110.710.1070.2030.2150.2390.4190", %"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187", %"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187", %"class.llvm::SmallVector.82.116.716.1076.2036.2156.2396.4196" }
%"class.llvm::SmallDenseMap.77.110.710.1070.2030.2150.2390.4190" = type { [4 x i8], i32, %"struct.llvm::AlignedCharArrayUnion.80.109.709.1069.2029.2149.2389.4189" }
%"struct.llvm::AlignedCharArrayUnion.80.109.709.1069.2029.2149.2389.4189" = type { %"struct.llvm::AlignedCharArray.81.108.708.1068.2028.2148.2388.4188" }
%"struct.llvm::AlignedCharArray.81.108.708.1068.2028.2148.2388.4188" = type { [384 x i8] }
%"class.llvm::SmallDenseMap.107.707.1067.2027.2147.2387.4187" = type { [4 x i8], i32, %"struct.llvm::AlignedCharArrayUnion.75.106.706.1066.2026.2146.2386.4186" }
%"struct.llvm::AlignedCharArrayUnion.75.106.706.1066.2026.2146.2386.4186" = type { %"struct.llvm::AlignedCharArray.76.105.705.1065.2025.2145.2385.4185" }
%"struct.llvm::AlignedCharArray.76.105.705.1065.2025.2145.2385.4185" = type { [256 x i8] }
%"class.llvm::SmallVector.82.116.716.1076.2036.2156.2396.4196" = type { %"class.llvm::SmallVectorImpl.83.114.714.1074.2034.2154.2394.4194", %"struct.llvm::SmallVectorStorage.87.115.715.1075.2035.2155.2395.4195" }
%"class.llvm::SmallVectorImpl.83.114.714.1074.2034.2154.2394.4194" = type { %"class.llvm::SmallVectorTemplateBase.84.113.713.1073.2033.2153.2393.4193" }
%"class.llvm::SmallVectorTemplateBase.84.113.713.1073.2033.2153.2393.4193" = type { %"class.llvm::SmallVectorTemplateCommon.85.112.712.1072.2032.2152.2392.4192" }
%"class.llvm::SmallVectorTemplateCommon.85.112.712.1072.2032.2152.2392.4192" = type { %"class.llvm::SmallVectorBase.24.624.984.1944.2064.2304.4104", %"struct.llvm::AlignedCharArrayUnion.86.111.711.1071.2031.2151.2391.4191" }
%"struct.llvm::AlignedCharArrayUnion.86.111.711.1071.2031.2151.2391.4191" = type { %"struct.llvm::AlignedCharArray.35.68.668.1028.1988.2108.2348.4148" }
%"struct.llvm::SmallVectorStorage.87.115.715.1075.2035.2155.2395.4195" = type { [127 x %"struct.llvm::AlignedCharArrayUnion.86.111.711.1071.2031.2151.2391.4191"] }
%"struct.std::pair.112.119.719.1079.2039.2159.2399.4199" = type { i32, %"struct.llvm::EVT.8.608.968.1928.2048.2288.4088" }
%"class.llvm::DenseMapBase.73.118.718.1078.2038.2158.2398.4198" = type { i8 }

@.str61 = external hidden unnamed_addr constant [80 x i8], align 1
@.str63 = external hidden unnamed_addr constant [80 x i8], align 1
@.str74 = external hidden unnamed_addr constant [49 x i8], align 1
@__PRETTY_FUNCTION__._ZN4llvm16DAGTypeLegalizer16GetWidenedVectorENS_7SDValueE = external hidden unnamed_addr constant [70 x i8], align 1
@.str98 = external hidden unnamed_addr constant [46 x i8], align 1
@__PRETTY_FUNCTION__._ZNK4llvm6SDNode12getValueTypeEj = external hidden unnamed_addr constant [57 x i8], align 1
@.str99 = external hidden unnamed_addr constant [19 x i8], align 1
@__PRETTY_FUNCTION__._ZN4llvm5SDLocC2EPKNS_6SDNodeE = external hidden unnamed_addr constant [41 x i8], align 1
@.str100 = external hidden unnamed_addr constant [50 x i8], align 1
@__PRETTY_FUNCTION__._ZNK4llvm6SDNode10getOperandEj = external hidden unnamed_addr constant [66 x i8], align 1

declare { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, i32 } @_ZN4llvm12SelectionDAG7getNodeEjNS_5SDLocENS_3EVTENS_7SDValueES3_(%"class.llvm::SelectionDAG.104.704.1064.2024.2144.2384.4184"*, i32, i8*, i32, i32, %"class.llvm::Type.7.607.967.1927.2047.2287.4087"*, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* byval align 8, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* byval align 8)

; Function Attrs: noreturn nounwind
declare void @__assert_fail(i8*, i8*, i32, i8*) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define hidden { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, i32 } @_ZN4llvm16DAGTypeLegalizer18WidenVecRes_BinaryEPNS_6SDNodeE(%"class.llvm::DAGTypeLegalizer.117.717.1077.2037.2157.2397.4197"* %this, %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"* %N) #2 align 2 {
entry:
  %Op.i43 = alloca %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083", align 8
  %ref.tmp.i = alloca %"struct.std::pair.112.119.719.1079.2039.2159.2399.4199", align 8
  %Op.i = alloca %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083", align 8
  %0 = bitcast %"struct.std::pair.112.119.719.1079.2039.2159.2399.4199"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %0) #1
  %retval.sroa.0.0.idx.i36 = getelementptr inbounds %"struct.std::pair.112.119.719.1079.2039.2159.2399.4199", %"struct.std::pair.112.119.719.1079.2039.2159.2399.4199"* %ref.tmp.i, i64 0, i32 1, i32 0, i32 0
  %retval.sroa.0.0.copyload.i37 = load i32, i32* %retval.sroa.0.0.idx.i36, align 8
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %0) #1
  %agg.tmp8.sroa.2.0.copyload = load i32, i32* undef, align 8
  %1 = bitcast %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* %Op.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %1) #1
  %2 = getelementptr %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083", %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* %Op.i, i64 0, i32 1
  store i32 %agg.tmp8.sroa.2.0.copyload, i32* %2, align 8

; CHECK: movl	(%rax), %eax
; CHECK-NOT: movl	%eax, {{[0-9]+}}(%rsp)
; CHECK: movl	[[OFF:[0-9]+]](%rsp), %r8d
; CHECK: movl	%eax, [[OFF]](%rsp)
; CHECK: movl	$-1, %ecx
; CHECK: callq	_ZN4llvm12SelectionDAG7getNodeEjNS_5SDLocENS_3EVTENS_7SDValueES3_

  %call18 = call { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, i32 } @_ZN4llvm12SelectionDAG7getNodeEjNS_5SDLocENS_3EVTENS_7SDValueES3_(%"class.llvm::SelectionDAG.104.704.1064.2024.2144.2384.4184"* undef, i32 undef, i8* undef, i32 -1, i32 %retval.sroa.0.0.copyload.i37, %"class.llvm::Type.7.607.967.1927.2047.2287.4087"* undef, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* byval align 8 undef, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* byval align 8 undef) #1
  ret { %"class.llvm::SDNode.10.610.970.1930.2050.2290.4090"*, i32 } %call18
}

; Function Attrs: nounwind uwtable
declare hidden %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* @_ZN4llvm12DenseMapBaseINS_13SmallDenseMapINS_7SDValueES2_Lj8ENS_12DenseMapInfoIS2_EEEES2_S2_S4_EixERKS2_(%"class.llvm::DenseMapBase.73.118.718.1078.2038.2158.2398.4198"*, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"* nocapture readonly) #2 align 2

declare hidden void @_ZN4llvm16DAGTypeLegalizer10RemapValueERNS_7SDValueE(%"class.llvm::DAGTypeLegalizer.117.717.1077.2037.2157.2397.4197"*, %"class.llvm::SDValue.3.603.963.1923.2043.2283.4083"*)

; Function Attrs: nounwind uwtable
declare hidden void @_ZNK4llvm18TargetLoweringBase17getTypeConversionERNS_11LLVMContextENS_3EVTE(%"struct.std::pair.112.119.719.1079.2039.2159.2399.4199"* noalias sret, %"class.llvm::TargetLoweringBase.50.650.1010.1970.2090.2330.4130"* readonly, %"class.llvm::LLVMContext.6.606.966.1926.2046.2286.4086"*, i32, %"class.llvm::Type.7.607.967.1927.2047.2287.4087"*) #2 align 2

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind uwtable }

