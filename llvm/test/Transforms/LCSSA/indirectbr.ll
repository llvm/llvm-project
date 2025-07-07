; RUN: opt < %s -passes=loop-simplify,lcssa -verify-loop-info -verify-dom-info -S | FileCheck %s

; LCSSA should work correctly in the case of an indirectbr that exits
; the loop, and the loop has exits with predecessors not within the loop
; (and btw these edges are unsplittable due to the indirectbr).
; PR5437
define i32 @test0(i1 %arg) nounwind {
; CHECK-LABEL: @test0
entry:
  br i1 %arg, label %"4", label %"3"

"3":                                              ; preds = %entry
  ret i32 0

"4":                                              ; preds = %entry
  br i1 %arg, label %"6", label %"5"

"5":                                              ; preds = %"4"
  unreachable

"6":                                              ; preds = %"4"
  br i1 %arg, label %"10", label %"13"

"10":                                             ; preds = %"6"
  br i1 %arg, label %"22", label %"15"

"13":                                             ; preds = %"6"
  unreachable

"15":                                             ; preds = %"23", %"10"
  unreachable

"22":                                             ; preds = %"10"
  br label %"23"

"23":                                             ; preds = %"1375", %"22"
  %0 = phi i32 [ undef, %"22" ], [ %1, %"1375" ]  ; <i32> [#uses=1]
  indirectbr ptr undef, [label %"15", label %"24", label %"25", label %"26", label %"27", label %"28", label %"29", label %"30", label %"32", label %"32", label %"33", label %"167", label %"173", label %"173", label %"173", label %"173", label %"173", label %"192", label %"193", label %"194", label %"196", label %"206", label %"231", label %"241", label %"251", label %"261", label %"307", label %"353", label %"354", label %"355", label %"361", label %"367", label %"400", label %"433", label %"466", label %"499", label %"509", label %"519", label %"529", label %"571", label %"589", label %"607", label %"635", label %"655", label %"664", label %"671", label %"680", label %"687", label %"692", label %"698", label %"704", label %"715", label %"715", label %"716", label %"725", label %"725", label %"725", label %"725", label %"724", label %"724", label %"724", label %"724", label %"737", label %"737", label %"737", label %"737", label %"761", label %"758", label %"759", label %"760", label %"766", label %"763", label %"764", label %"765", label %"771", label %"768", label %"769", label %"770", label %"780", label %"777", label %"778", label %"779", label %"821", label %"826", label %"831", label %"832", label %"833", label %"836", label %"836", label %"886", label %"905", label %"978", label %"978", label %"1136", label %"1166", label %"1179", label %"1201", label %"1212", label %"1212", label %"1274", label %"1284", label %"1284", label %"1346", label %"1347", label %"1348", label %"1349", label %"1350", label %"1353", label %"1353", label %"1353", label %"1355", label %"1355", label %"1357", label %"1357", label %"1358", label %"1359", label %"1374", label %"1375", label %"1376", label %"1377", label %"1378", label %"1379", label %"1386", label %"1395", label %"1394", label %"1425", label %"1426", label %"1440", label %"1449", label %"1455", label %"1461", label %"1471", label %"1482", label %"1484", label %"1486", label %"1489", label %"1489", label %"1492", label %"1494", label %"1494", label %"1497", label %"1499", label %"1499", label %"1515", label %"1546", label %"1546", label %"1566", label %"1584", label %"1587", label %"1591", label %"1605", label %"1609", label %"1609", label %"1640", label %"1648", label %"1651", label %"1703", label %"1710", label %"1718", label %"1724", label %"1725", label %"1726", label %"1727", label %"1728", label %"1731", label %"1732", label %"1733", label %"1734", label %"1735", label %"1741", label %"1750", label %"1752", label %"1754", label %"1755", label %"1757", label %"1759", label %"1761", label %"1764", label %"1764", label %"1766", label %"1768", label %"1775", label %"1775", label %"1781", label %"1781", label %"1790", label %"1827", label %"1836", label %"1836", label %"1845", label %"1845", label %"1848", label %"1849", label %"1851", label %"1853", label %"1856", label %"1861", label %"1861"]

"24":                                             ; preds = %"23"
  unreachable

"25":                                             ; preds = %"23"
  unreachable

"26":                                             ; preds = %"23"
  unreachable

"27":                                             ; preds = %"23"
  unreachable

"28":                                             ; preds = %"23"
  unreachable

"29":                                             ; preds = %"23"
  unreachable

"30":                                             ; preds = %"23"
  unreachable

"32":                                             ; preds = %"23", %"23"
  unreachable

"33":                                             ; preds = %"23"
  unreachable

"167":                                            ; preds = %"23"
  unreachable

"173":                                            ; preds = %"23", %"23", %"23", %"23", %"23"
  unreachable

"192":                                            ; preds = %"23"
  unreachable

"193":                                            ; preds = %"23"
  unreachable

"194":                                            ; preds = %"23"
  unreachable

"196":                                            ; preds = %"23"
  unreachable

"206":                                            ; preds = %"23"
  unreachable

"231":                                            ; preds = %"23"
  unreachable

"241":                                            ; preds = %"23"
  unreachable

"251":                                            ; preds = %"23"
  unreachable

"261":                                            ; preds = %"23"
  unreachable

"307":                                            ; preds = %"23"
  unreachable

"353":                                            ; preds = %"23"
  unreachable

"354":                                            ; preds = %"23"
  unreachable

"355":                                            ; preds = %"23"
  unreachable

"361":                                            ; preds = %"23"
  unreachable

"367":                                            ; preds = %"23"
  unreachable

"400":                                            ; preds = %"23"
  unreachable

"433":                                            ; preds = %"23"
  unreachable

"466":                                            ; preds = %"23"
  unreachable

"499":                                            ; preds = %"23"
  unreachable

"509":                                            ; preds = %"23"
  unreachable

"519":                                            ; preds = %"23"
  unreachable

"529":                                            ; preds = %"23"
  unreachable

"571":                                            ; preds = %"23"
  unreachable

"589":                                            ; preds = %"23"
  unreachable

"607":                                            ; preds = %"23"
  unreachable

"635":                                            ; preds = %"23"
  unreachable

"655":                                            ; preds = %"23"
  unreachable

"664":                                            ; preds = %"23"
  unreachable

"671":                                            ; preds = %"23"
  unreachable

"680":                                            ; preds = %"23"
  unreachable

"687":                                            ; preds = %"23"
  unreachable

"692":                                            ; preds = %"23"
  br label %"1862"

"698":                                            ; preds = %"23"
  unreachable

"704":                                            ; preds = %"23"
  unreachable

"715":                                            ; preds = %"23", %"23"
  unreachable

"716":                                            ; preds = %"23"
  unreachable

"724":                                            ; preds = %"23", %"23", %"23", %"23"
  unreachable

"725":                                            ; preds = %"23", %"23", %"23", %"23"
  unreachable

"737":                                            ; preds = %"23", %"23", %"23", %"23"
  unreachable

"758":                                            ; preds = %"23"
  unreachable

"759":                                            ; preds = %"23"
  unreachable

"760":                                            ; preds = %"23"
  unreachable

"761":                                            ; preds = %"23"
  unreachable

"763":                                            ; preds = %"23"
  unreachable

"764":                                            ; preds = %"23"
  unreachable

"765":                                            ; preds = %"23"
  br label %"766"

"766":                                            ; preds = %"765", %"23"
  unreachable

"768":                                            ; preds = %"23"
  unreachable

"769":                                            ; preds = %"23"
  unreachable

"770":                                            ; preds = %"23"
  unreachable

"771":                                            ; preds = %"23"
  unreachable

"777":                                            ; preds = %"23"
  unreachable

"778":                                            ; preds = %"23"
  unreachable

"779":                                            ; preds = %"23"
  unreachable

"780":                                            ; preds = %"23"
  unreachable

"821":                                            ; preds = %"23"
  unreachable

"826":                                            ; preds = %"23"
  unreachable

"831":                                            ; preds = %"23"
  unreachable

"832":                                            ; preds = %"23"
  unreachable

"833":                                            ; preds = %"23"
  unreachable

"836":                                            ; preds = %"23", %"23"
  unreachable

"886":                                            ; preds = %"23"
  unreachable

"905":                                            ; preds = %"23"
  unreachable

"978":                                            ; preds = %"23", %"23"
  unreachable

"1136":                                           ; preds = %"23"
  unreachable

"1166":                                           ; preds = %"23"
  unreachable

"1179":                                           ; preds = %"23"
  unreachable

"1201":                                           ; preds = %"23"
  unreachable

"1212":                                           ; preds = %"23", %"23"
  unreachable

"1274":                                           ; preds = %"23"
  unreachable

"1284":                                           ; preds = %"23", %"23"
  unreachable

"1346":                                           ; preds = %"23"
  unreachable

"1347":                                           ; preds = %"23"
  unreachable

"1348":                                           ; preds = %"23"
  unreachable

"1349":                                           ; preds = %"23"
  unreachable

"1350":                                           ; preds = %"23"
  unreachable

"1353":                                           ; preds = %"23", %"23", %"23"
  unreachable

"1355":                                           ; preds = %"23", %"23"
  unreachable

"1357":                                           ; preds = %"23", %"23"
  unreachable

"1358":                                           ; preds = %"23"
  unreachable

"1359":                                           ; preds = %"23"
  unreachable

"1374":                                           ; preds = %"23"
  unreachable

"1375":                                           ; preds = %"23"
  %1 = zext i8 undef to i32                       ; <i32> [#uses=1]
  br label %"23"

"1376":                                           ; preds = %"23"
  unreachable

"1377":                                           ; preds = %"23"
  unreachable

"1378":                                           ; preds = %"23"
  unreachable

"1379":                                           ; preds = %"23"
  unreachable

"1386":                                           ; preds = %"23"
  unreachable

"1394":                                           ; preds = %"23"
  unreachable

"1395":                                           ; preds = %"23"
  unreachable

"1425":                                           ; preds = %"23"
  unreachable

"1426":                                           ; preds = %"23"
  unreachable

"1440":                                           ; preds = %"23"
  unreachable

"1449":                                           ; preds = %"23"
  unreachable

"1455":                                           ; preds = %"23"
  unreachable

"1461":                                           ; preds = %"23"
  unreachable

"1471":                                           ; preds = %"23"
  unreachable

"1482":                                           ; preds = %"23"
  unreachable

"1484":                                           ; preds = %"23"
  unreachable

"1486":                                           ; preds = %"23"
  unreachable

"1489":                                           ; preds = %"23", %"23"
  unreachable

"1492":                                           ; preds = %"23"
  unreachable

"1494":                                           ; preds = %"23", %"23"
  unreachable

"1497":                                           ; preds = %"23"
  unreachable

"1499":                                           ; preds = %"23", %"23"
  unreachable

"1515":                                           ; preds = %"23"
  unreachable

"1546":                                           ; preds = %"23", %"23"
  unreachable

"1566":                                           ; preds = %"23"
  br i1 %arg, label %"1569", label %"1568"

"1568":                                           ; preds = %"1566"
  unreachable

"1569":                                           ; preds = %"1566"
  unreachable

"1584":                                           ; preds = %"23"
  unreachable

"1587":                                           ; preds = %"23"
  unreachable

"1591":                                           ; preds = %"23"
  unreachable

"1605":                                           ; preds = %"23"
  unreachable

"1609":                                           ; preds = %"23", %"23"
  unreachable

"1640":                                           ; preds = %"23"
  unreachable

"1648":                                           ; preds = %"23"
  unreachable

"1651":                                           ; preds = %"23"
  unreachable

"1703":                                           ; preds = %"23"
  unreachable

"1710":                                           ; preds = %"23"
  unreachable

"1718":                                           ; preds = %"23"
  unreachable

"1724":                                           ; preds = %"23"
  unreachable

"1725":                                           ; preds = %"23"
  unreachable

"1726":                                           ; preds = %"23"
  unreachable

"1727":                                           ; preds = %"23"
  unreachable

"1728":                                           ; preds = %"23"
  unreachable

"1731":                                           ; preds = %"23"
  unreachable

"1732":                                           ; preds = %"23"
  unreachable

"1733":                                           ; preds = %"23"
  unreachable

"1734":                                           ; preds = %"23"
  unreachable

"1735":                                           ; preds = %"23"
  unreachable

"1741":                                           ; preds = %"23"
  unreachable

"1750":                                           ; preds = %"23"
  unreachable

"1752":                                           ; preds = %"23"
  unreachable

"1754":                                           ; preds = %"23"
  unreachable

"1755":                                           ; preds = %"23"
  unreachable

"1757":                                           ; preds = %"23"
  unreachable

"1759":                                           ; preds = %"23"
  unreachable

"1761":                                           ; preds = %"23"
  unreachable

"1764":                                           ; preds = %"23", %"23"
  %2 = icmp eq i32 %0, 168                        ; <i1> [#uses=0]
  unreachable

"1766":                                           ; preds = %"23"
  unreachable

"1768":                                           ; preds = %"23"
  unreachable

"1775":                                           ; preds = %"23", %"23"
  unreachable

"1781":                                           ; preds = %"23", %"23"
  unreachable

"1790":                                           ; preds = %"23"
  unreachable

"1827":                                           ; preds = %"23"
  unreachable

"1836":                                           ; preds = %"23", %"23"
  br label %"1862"

"1845":                                           ; preds = %"23", %"23"
  unreachable

"1848":                                           ; preds = %"23"
  unreachable

"1849":                                           ; preds = %"23"
  unreachable

"1851":                                           ; preds = %"23"
  unreachable

"1853":                                           ; preds = %"23"
  unreachable

"1856":                                           ; preds = %"23"
  unreachable

"1861":                                           ; preds = %"23", %"23"
  unreachable

"41":                                             ; preds = %"23", %"23"
  unreachable

"1862":                                           ; preds = %"1836", %"692"
  unreachable
}

; An exit for Loop L1 may be the header of a disjoint Loop L2.  Thus, when we
; create PHIs in one of such exits we are also inserting PHIs in L2 header. This
; could break LCSSA form for L2 because these inserted PHIs can also have uses
; in L2 exits. Test that we don't assert/crash on that.
define void @test1(i1 %arg) {
; CHECK-LABEL: @test1
  br label %lab1

lab1:
  %tmp21 = add i32 undef, 677038203
  br i1 %arg, label %lab2, label %exit

lab2:
  indirectbr ptr undef, [label %lab1, label %lab3]

lab3:
; CHECK: %tmp21.lcssa1 = phi i32 [ %tmp21.lcssa1, %lab4 ], [ %tmp21, %lab2 ]
  %tmp12 = phi i32 [ %tmp21, %lab2 ], [ %tmp12, %lab4 ]
  br i1 %arg, label %lab5, label %lab4

lab4:
  br label %lab3

lab5:
; CHECK:  %tmp21.lcssa1.lcssa = phi i32 [ %tmp21.lcssa1, %lab3 ]
  %tmp15 = add i32 %tmp12, undef
  br label %exit

exit:
  ret void
}
