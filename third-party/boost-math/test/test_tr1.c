/*  (C) Copyright John Maddock 2008.
  Use, modification and distribution are subject to the
  Boost Software License, Version 1.0. (See accompanying file
  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
  */

#include <math.h>
#include <float.h>
#include <stdio.h>

#include <boost/math/tr1.hpp>

unsigned errors = 0;

void check_close_f(float v1, float v2, float tol, int line)
{
   float err;
   if((v1 == 0) || (v2 ==0))
   {
      err = fabsf(v1 - v2);
   }
   else
   {
      err = fabsf((v1 - v2) / v2);
   }
   if(err * 100 > tol)
   {
      errors += 1;
      printf("Error at line %d, with error %e (max allowed %e)\n", line, (double)err, (double)(tol / 100));
   }
}
void check_close(double v1, double v2, double tol, int line)
{
   double err;
   if((v1 == 0) || (v2 ==0))
   {
      err = fabs(v1 - v2);
   }
   else
   {
      err = fabs((v1 - v2) / v2);
   }
   if(err * 100 > tol)
   {
      errors += 1;
      printf("Error at line %d, with error %e (max allowed %e)\n", line, (double)err, (double)(tol / 100));
   }
}
void check_close_l(long double v1, long double v2, long double tol, int line)
{
   long double err;
   if((v1 == 0) || (v2 ==0))
   {
      err = fabsl(v1 - v2);
   }
   else
   {
      err = fabsl((v1 - v2) / v2);
   }
   if(err * 100 > tol)
   {
      errors += 1;
      printf("Error at line %d, with error %e (max allowed %e)\n", line, (double)err, (double)(tol / 100));
   }
}

void check(int b, int line)
{
   if(b == 0)
   {
      errors += 1;
      printf("Error at line %d\n", line);
   }
}

void check_close_fraction_f(float v1, float v2, float tol, int line)
{
   check_close_f(v1, v2, tol * 100, line);
}
void check_close_fraction(double v1, double v2, double tol, int line)
{
   check_close(v1, v2, tol * 100, line);
}
void check_close_fraction_l(long double v1, long double v2, long double tol, int line)
{
   check_close_l(v1, v2, tol * 100, line);
}

void test_values_f(const char* name)
{
#ifndef TEST_LD
   //
   // First the C99 math functions:
   //
   float eps = FLT_EPSILON;
   float sv;
   check_close_f(acoshf(coshf(0.5f)), 0.5f, 5000 * eps, __LINE__);
   check_close_f(asinhf(sinhf(0.5f)), 0.5f, 5000 * eps, __LINE__);
   check_close_f(atanhf(tanhf(0.5f)), 0.5f, 5000 * eps, __LINE__);

   check_close_f(cbrtf(1.5f * 1.5f * 1.5f), 1.5f, 5000 * eps, __LINE__);

   check(copysignf(1.0f, 1.0f) == 1.0f, __LINE__);
   check(copysignf(1.0f, -1.0f) == -1.0f, __LINE__);
   check(copysignf(-1.0f, 1.0f) == 1.0f, __LINE__);
   check(copysignf(-1.0f, -1.0f) == -1.0f, __LINE__);

   check_close_f(erfcf(0.125), 0.85968379519866618260697055347837660181302041685015f, eps * 1000, __LINE__);
   check_close_f(erfcf(0.5), 0.47950012218695346231725334610803547126354842424204f, eps * 1000, __LINE__);
   check_close_f(erfcf(1), 0.15729920705028513065877936491739074070393300203370f, eps * 1000, __LINE__);
   check_close_f(erfcf(5), 1.5374597944280348501883434853833788901180503147234e-12f, eps * 1000, __LINE__);
   check_close_f(erfcf(-0.125), 1.1403162048013338173930294465216233981869795831498f, eps * 1000, __LINE__);
   check_close_f(erfcf(-0.5), 1.5204998778130465376827466538919645287364515757580f, eps * 1000, __LINE__);
   check_close_f(erfcf(0), 1, eps * 1000, __LINE__);

   check_close_f(erff(0.125), 0.14031620480133381739302944652162339818697958314985f, eps * 1000, __LINE__);
   check_close_f(erff(0.5), 0.52049987781304653768274665389196452873645157575796f, eps * 1000, __LINE__);
   check_close_f(erff(1), 0.84270079294971486934122063508260925929606699796630f, eps * 1000, __LINE__);
   check_close_f(erff(5), 0.9999999999984625402055719651498116565146166211099f, eps * 1000, __LINE__);
   check_close_f(erff(-0.125), -0.14031620480133381739302944652162339818697958314985f, eps * 1000, __LINE__);
   check_close_f(erff(-0.5), -0.52049987781304653768274665389196452873645157575796f, eps * 1000, __LINE__);
   check_close_f(erff(0), 0, eps * 1000, __LINE__);

   check_close_f(log1pf(0.582029759883880615234375e0), 0.4587086807259736626531803258754840111707e0f, eps * 1000, __LINE__);
   check_close_f(expm1f(0.582029759883880615234375e0), 0.7896673415707786528734865994546559029663e0f, eps * 1000, __LINE__);
   check_close_f(log1pf(-0.2047410048544406890869140625e-1), -0.2068660038044094868521052319477265955827e-1f, eps * 1000, __LINE__);
   check_close_f(expm1f(-0.2047410048544406890869140625e-1), -0.2026592921724753704129022027337835687888e-1f, eps * 1000, __LINE__);

   check_close_f(fmaxf(0.1f, -0.1f), 0.1f, 0, __LINE__);
   check_close_f(fminf(0.1f, -0.1f), -0.1f, 0, __LINE__);

   check_close_f(hypotf(1.0f, 3.0f), sqrtf(10.0f), eps * 500, __LINE__);

   check_close_f(lgammaf(3.5), 1.2009736023470742248160218814507129957702389154682f, 5000 * eps, __LINE__);
   check_close_f(lgammaf(0.125), 2.0194183575537963453202905211670995899482809521344f, 5000 * eps, __LINE__);
   check_close_f(lgammaf(-0.125), 2.1653002489051702517540619481440174064962195287626f, 5000 * eps, __LINE__);
   check_close_f(lgammaf(-3.125), 0.1543111276840418242676072830970532952413339012367f, 5000 * eps, __LINE__);
   check_close_f(lgammaf(-53249.0/1024), -149.43323093420259741100038126078721302600128285894f, 5000 * eps, __LINE__);

   check_close_f(tgammaf(3.5), 3.3233509704478425511840640312646472177454052302295f, 5000 * eps, __LINE__);
   check_close_f(tgammaf(0.125), 7.5339415987976119046992298412151336246104195881491f, 5000 * eps, __LINE__);
   check_close_f(tgammaf(-0.125), -8.7172188593831756100190140408231437691829605421405f, 5000 * eps, __LINE__);
   check_close_f(tgammaf(-3.125), 1.1668538708507675587790157356605097019141636072094f, 5000 * eps, __LINE__);

   check(llroundf(2.5f) == 3ll, __LINE__);
   check(llroundf(2.25f) == 2ll, __LINE__);

   check(lroundf(2.5f) == 3.0f, __LINE__);
   check(lroundf(2.25f) == 2.0f, __LINE__);
   check(roundf(2.5f) == 3.0f, __LINE__);
   check(roundf(2.25f) == 2.0f, __LINE__);

   check(nextafterf(1.0f, 2.0f) > 1.0f, __LINE__);
   check(nextafterf(1.0f, -2.0f) < 1.0f, __LINE__);
   check(nextafterf(nextafterf(1.0f, 2.0f), -2.0f) == 1.0f, __LINE__);
   check(nextafterf(nextafterf(1.0f, -2.0f), 2.0f) == 1.0f, __LINE__);
   check(nextafterf(1.0f, 2.0f) > 1.0f, __LINE__);
   check(nextafterf(1.0f, -2.0f) < 1.0f, __LINE__);
   check(nextafterf(nextafterf(1.0f, 2.0f), -2.0f) == 1.0f, __LINE__);
   check(nextafterf(nextafterf(1.0f, -2.0f), 2.0f) == 1.0f, __LINE__);

   check(truncf(2.5f) == 2.0f, __LINE__);
   check(truncf(2.25f) == 2.0f, __LINE__);

   //
   // Now for the TR1 math functions:
   //
   check_close_fraction_f(assoc_laguerref(4, 5, 0.5f), 88.31510416666666666666666666666666666667f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_laguerref(10, 0, 2.5f), -0.8802526766660982969576719576719576719577f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_laguerref(10, 1, 4.5f), 1.564311458042689732142857142857142857143f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_laguerref(10, 6, 8.5f), 20.51596541066649098875661375661375661376f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_laguerref(10, 12, 12.5f), -199.5560968456234671241181657848324514991f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_laguerref(50, 40, 12.5f), -4.996769495006119488583146995907246595400e16f, eps * 100, __LINE__);

   check_close_fraction_f(laguerref(1, 0.5f), 0.5f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(4, 0.5f), -0.3307291666666666666666666666666666666667f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(7, 0.5f), -0.5183392237103174603174603174603174603175f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(20, 0.5f), 0.3120174870800154148915399248893113634676f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(50, 0.5f), -0.3181388060269979064951118308575628226834f, eps * 100, __LINE__);

   check_close_fraction_f(laguerref(1, -0.5f), 1.5f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(4, -0.5f), 3.835937500000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(7, -0.5f), 7.950934709821428571428571428571428571429f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(20, -0.5f), 76.12915699869631476833699787070874048223f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(50, -0.5f), 2307.428631277506570629232863491518399720f, eps * 100, __LINE__);

   check_close_fraction_f(laguerref(1, 4.5f), -3.500000000000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(4, 4.5f), 0.08593750000000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(7, 4.5f), -1.036928013392857142857142857142857142857f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(20, 4.5f), 1.437239150257817378525582974722170737587f, eps * 100, __LINE__);
   check_close_fraction_f(laguerref(50, 4.5f), -0.7795068145562651416494321484050019245248f, eps * 100, __LINE__);

   check_close_fraction_f(assoc_legendref(4, 2, 0.5f), 4.218750000000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_legendref(7, 5, 0.5f), 5696.789530152175143607977274672800795328f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_legendref(4, 2, -0.5f), 4.218750000000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(assoc_legendref(7, 5, -0.5f), 5696.789530152175143607977274672800795328f, eps * 100, __LINE__);

   check_close_fraction_f(legendref(1, 0.5f), 0.5f, eps * 100, __LINE__);
   check_close_fraction_f(legendref(4, 0.5f), -0.2890625000000000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(legendref(7, 0.5f), 0.2231445312500000000000000000000000000000f, eps * 100, __LINE__);
   check_close_fraction_f(legendref(40, 0.5f), -0.09542943523261546936538467572384923220258f, eps * 100, __LINE__);

   sv = eps / 1024;
   check_close_f(betaf(1, 1), 1, eps * 20 * 100, __LINE__);
   check_close_f(betaf(1, 4), 0.25, eps * 20 * 100, __LINE__);
   check_close_f(betaf(4, 1), 0.25, eps * 20 * 100, __LINE__);
   check_close_f(betaf(sv, 4), 1/sv, eps * 20 * 100, __LINE__);
   check_close_f(betaf(4, sv), 1/sv, eps * 20 * 100, __LINE__);
   check_close_f(betaf(4, 20), 0.00002823263692828910220214568040654997176736f, eps * 20 * 100, __LINE__);
   check_close_f(betaf(0.0125f, 0.000023f), 43558.24045647538375006349016083320744662f, eps * 20 * 100, __LINE__);

   check_close_f(comp_ellint_1f(0), 1.5707963267948966192313216916397514420985846996876f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(0.125), 1.5769867712158131421244030532288080803822271060839f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(0.25), 1.5962422221317835101489690714979498795055744578951f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(300/1024.0f), 1.6062331054696636704261124078746600894998873503208f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(400/1024.0f), 1.6364782007562008756208066125715722889067992997614f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(-0.5), 1.6857503548125960428712036577990769895008008941411f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_1f(-0.75), 1.9109897807518291965531482187613425592531451316788f, eps * 5000, __LINE__);

   check_close_f(comp_ellint_2f(-1), 1.0f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(0), 1.5707963267948966192313216916397514420985846996876f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(100 / 1024.0f), 1.5670445330545086723323795143598956428788609133377f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(200 / 1024.0f), 1.5557071588766556854463404816624361127847775545087f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(300 / 1024.0f), 1.5365278991162754883035625322482669608948678755743f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(400 / 1024.0f), 1.5090417763083482272165682786143770446401437564021f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(-0.5), 1.4674622093394271554597952669909161360253617523272f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(-600 / 1024.0f), 1.4257538571071297192428217218834579920545946473778f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(-800 / 1024.0f), 1.2927868476159125056958680222998765985004489572909f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_2f(-900 / 1024.0f), 1.1966864890248739524112920627353824133420353430982f, eps * 5000, __LINE__);

   check_close_f(comp_ellint_3f(0.2f, 0), 1.586867847454166237308008033828114192951f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(0.4f, 0), 1.639999865864511206865258329748601457626f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(0.0f, 0), 1.57079632679489661923132169163975144209858469968755291048747f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(0.0f, 0.5), 2.221441469079183123507940495030346849307f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(0.3f, -4), 0.712708870925620061597924858162260293305195624270730660081949f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(-0.5f, -1e+05), 0.00496944596485066055800109163256108604615568144080386919012831f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(-0.75f, -1e+10), 0.0000157080225184890546939710019277357161497407143903832703317801f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(-0.875f, 1 / 1024.0f), 2.18674503176462374414944618968850352696579451638002110619287f, eps * 5000, __LINE__);
   check_close_f(comp_ellint_3f(-0.875f, 1023/1024.0f), 101.045289804941384100960063898569538919135722087486350366997f, eps * 5000, __LINE__);

   check_close_f(cyl_bessel_if(2.25f, 1/(1024.0f*1024.0f)), 2.34379212133481347189068464680335815256364262507955635911656e-15f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(5.5f, 3.125), 0.0583514045989371500460946536220735787163510569634133670181210f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(-5 + 1.0f/1024.0f, 2.125), 0.0267920938009571023702933210070984416052633027166975342895062f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(-5.5f, 10), 597.577606961369169607937419869926705730305175364662688426534f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(-10486074.0f/(1024.0f*1024), 1/1024.0f), 1.41474005665181350367684623930576333542989766867888186478185e35f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(-10486074.0f/(1024.0f*1024), 50), 1.07153277202900671531087024688681954238311679648319534644743e20f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(144794.0f/1024.0f, 100), 2066.27694757392660413922181531984160871678224178890247540320f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_if(-144794.0f/1024.0f, 100), 2066.27694672763190927440969155740243346136463461655104698748f, eps * 5000, __LINE__);

   check_close_f(cyl_bessel_jf(2457/1024.0f, 1/1024.0f), 3.80739920118603335646474073457326714709615200130620574875292e-9f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(5.5f, 3217.0f/1024), 0.0281933076257506091621579544064767140470089107926550720453038f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-5.5f, 3217.0f/1024), -2.55820064470647911823175836997490971806135336759164272675969f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-5.5f, 1e+04), 2.449843111985605522111159013846599118397e-03f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(5.5f, 1e+04), 0.00759343502722670361395585198154817047185480147294665270646578f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(5.5f, 1e+06), -0.000747424248595630177396350688505919533097973148718960064663632f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(5.125f, 1e+06), -0.000776600124835704280633640911329691642748783663198207360238214f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(5.875f, 1e+06), -0.000466322721115193071631008581529503095819705088484386434589780f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(0.5f, 101), 0.0358874487875643822020496677692429287863419555699447066226409f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-5.5f, 1e+04), 0.00244984311198560552211115901384659911839737686676766460822577f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-5.5f, 1e+06), 0.000279243200433579511095229508894156656558211060453622750659554f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-0.5f, 101), 0.0708184798097594268482290389188138201440114881159344944791454f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-10486074 / (1024*1024.0f), 1/1024.0f), 1.41474013160494695750009004222225969090304185981836460288562e35f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-10486074 / (1024*1024.0f), 15), -0.0902239288885423309568944543848111461724911781719692852541489f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(10486074 / (1024*1024.0f), 1e+02f), -0.0547064914615137807616774867984047583596945624129838091326863f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(10486074 / (1024*1024.0f), 2e+04f), -0.00556783614400875611650958980796060611309029233226596737701688f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_jf(-10486074 / (1024*1024.0f), 1e+02f), -0.0547613660316806551338637153942604550779513947674222863858713f, eps * 5000, __LINE__);

   check_close_f(cyl_bessel_kf(0.5f, 0.875), 0.558532231646608646115729767013630967055657943463362504577189f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(0.5f, 1.125), 0.383621010650189547146769320487006220295290256657827220786527f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(2.25f, ldexpf(1.0f, -30)), 5.62397392719283271332307799146649700147907612095185712015604e20f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(5.5f, 3217/1024.0f), 1.30623288775012596319554857587765179889689223531159532808379f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(-5.5f, 10), 0.0000733045300798502164644836879577484533096239574909573072142667f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(-5.5f, 100), 5.41274555306792267322084448693957747924412508020839543293369e-45f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(10240/1024.0f, 1/1024.0f), 2.35522579263922076203415803966825431039900000000993410734978e38f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(10240/1024.0f, 10), 0.00161425530039067002345725193091329085443750382929208307802221f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(144793/1024.0f, 100), 1.39565245860302528069481472855619216759142225046370312329416e-6f, eps * 5000, __LINE__);
   check_close_f(cyl_bessel_kf(144793/1024.0f, 200), 0.0f, eps * 5000, __LINE__);

   check_close_f(cyl_neumannf(0.5f, 1 / (1024.0f*1024)), -817.033790261762580469303126467917092806755460418223776544122f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(5.5f, 3.125), -2.61489440328417468776474188539366752698192046890955453259866f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(-5.5f, 3.125), -0.0274994493896489729948109971802244976377957234563871795364056f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(-5.5f, 1e+04), -0.00759343502722670361395585198154817047185480147294665270646578f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(-10486074 / (1024*1024.0f), 1/1024.0f), -1.50382374389531766117868938966858995093408410498915220070230e38f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(-10486074 / (1024*1024.0f), 1e+02f), 0.0583041891319026009955779707640455341990844522293730214223545f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(141.75f, 1e+02), -5.38829231428696507293191118661269920130838607482708483122068e9f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(141.75f, 2e+04), -0.00376577888677186194728129112270988602876597726657372330194186f, eps * 5000, __LINE__);
   check_close_f(cyl_neumannf(-141.75f, 1e+02), -3.81009803444766877495905954105669819951653361036342457919021e9f, eps * 5000, __LINE__);

   check_close_f(ellint_1f(0, 0), 0, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0, -10), -10, eps * 5000, __LINE__);
   check_close_f(ellint_1f(-1, -1), -1.2261911708835170708130609674719067527242483502207f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.875f, -4), -5.3190556182262405182189463092940736859067548232647f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(-0.625f, 8), 9.0419973860310100524448893214394562615252527557062f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.875f, 1e-05f), 0.000010000000000127604166668510945638036143355898993088f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(10/1024.0f, 1e+05f), 100002.38431454899771096037307519328741455615271038f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(1, 1e-20f), 1.0000000000000000000000000000000000000000166666667e-20f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(1e-20f, 1e-20f), 1.000000000000000e-20f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(400/1024.0f, 1e+20f), 1.0418143796499216839719289963154558027005142709763e20f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, 2), 2.1765877052210673672479877957388515321497888026770f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, 4), 4.2543274975235836861894752787874633017836785640477f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, 6), 6.4588766202317746302999080620490579800463614807916f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, 10), 10.697409951222544858346795279378531495869386960090f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, -2), -2.1765877052210673672479877957388515321497888026770f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, -4), -4.2543274975235836861894752787874633017836785640477f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, -6), -6.4588766202317746302999080620490579800463614807916f, eps * 5000, __LINE__);
   check_close_f(ellint_1f(0.5f, -10), -10.697409951222544858346795279378531495869386960090f, eps * 5000, __LINE__);

   check_close_f(ellint_2f(0, 0), 0, eps * 5000, __LINE__);
   check_close_f(ellint_2f(0, -10), -10, eps * 5000, __LINE__);
   check_close_f(ellint_2f(-1, -1), -0.84147098480789650665250232163029899962256306079837f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(900 / 1024.0f, -4), -3.1756145986492562317862928524528520686391383168377f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(-600 / 1024.0f, 8), 7.2473147180505693037677015377802777959345489333465f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(800 / 1024.0f, 1e-05f), 9.999999999898274739584436515967055859383969942432E-6f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(100 / 1024.0f, 1e+05f), 99761.153306972066658135668386691227343323331995888f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(-0.5f, 1e+10f), 9.3421545766487137036576748555295222252286528414669e9f, eps * 5000, __LINE__);
   check_close_f(ellint_2f(400 / 1024.0f, ldexpf(1, 66)), 7.0886102721911705466476846969992069994308167515242e19f, eps * 5000, __LINE__);

   check_close_f(ellint_3f(0, 1, -1), -1.557407724654902230506974807458360173087f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.4f, 0, -4), -4.153623371196831087495427530365430979011f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(-0.6f, 0, 8), 8.935930619078575123490612395578518914416f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.25f, 0, 0.5f), 0.501246705365439492445236118603525029757890291780157969500480f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 0, 0.5f), 0.5f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, -2, 0.5f), 0.437501067017546278595664813509803743009132067629603474488486f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 0.25f, 0.5f), 0.510269830229213412212501938035914557628394166585442994564135f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 0.75f, 0.5f), 0.533293253875952645421201146925578536430596894471541312806165f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 0.75f, 0.75), 0.871827580412760575085768367421866079353646112288567703061975f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 1, 0.25f), 0.255341921221036266504482236490473678204201638800822621740476f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 2, 0.25f), 0.261119051639220165094943572468224137699644963125853641716219f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0, 1023/1024.0f, 1.5), 13.2821612239764190363647953338544569682942329604483733197131f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.5f, 0.5f, -1), -1.228014414316220642611298946293865487807f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.5f, 0.5f, 1e+10f), 1.536591003599172091573590441336982730551e+10f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.75f, -1e+05f, 10), 0.0347926099493147087821620459290460547131012904008557007934290f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.875f, -1e+10f, 10), 0.000109956202759561502329123384755016959364346382187364656768212f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.875f, -1e+10f, 1e+20f), 1.00000626665567332602765201107198822183913978895904937646809e15f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.875f, -1e+10f, 1608/1024.0f), 0.0000157080616044072676127333183571107873332593142625043567690379f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.875f, 1-1 / 1024.0f, 1e+20f), 6.43274293944380717581167058274600202023334985100499739678963e21f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.25f, 50, 0.1f), 0.124573770342749525407523258569507331686458866564082916835900f, eps * 5000, __LINE__);
   check_close_f(ellint_3f(0.25f, 1.125f, 1), 1.77299767784815770192352979665283069318388205110727241629752f, eps * 5000, __LINE__);

   check_close_f(expintf(1/1024.0f), -6.35327933972759151358547423727042905862963067106751711596065f, eps * 5000, __LINE__);
   check_close_f(expintf(0.125), -1.37320852494298333781545045921206470808223543321810480716122f, eps * 5000, __LINE__);
   check_close_f(expintf(0.5), 0.454219904863173579920523812662802365281405554352642045162818f, eps * 5000, __LINE__);
   check_close_f(expintf(1), 1.89511781635593675546652093433163426901706058173270759164623f, eps * 5000, __LINE__);
   check_close_f(expintf(50.5), 1.72763195602911805201155668940185673806099654090456049881069e20f, eps * 5000, __LINE__);
   check_close_f(expintf(-1/1024.0f), -6.35523246483107180261445551935803221293763008553775821607264f, eps * 5000, __LINE__);
   check_close_f(expintf(-0.125), -1.62342564058416879145630692462440887363310605737209536579267f, eps * 5000, __LINE__);
   check_close_f(expintf(-0.5), -0.559773594776160811746795939315085235226846890316353515248293f, eps * 5000, __LINE__);
   check_close_f(expintf(-1), -0.219383934395520273677163775460121649031047293406908207577979f, eps * 5000, __LINE__);
   check_close_f(expintf(-50.5), -2.27237132932219350440719707268817831250090574830769670186618e-24f, eps * 5000, __LINE__);

   check_close_fraction_f(hermitef(0, 1), 1.L, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(1, 1), 2.L, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(1, 2), 4.L, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(1, 10), 20, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(1, 100), 200, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(1, 1e6), 2e6f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(10, 30), 5.896624628001300E+17f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(10, 1000), 1.023976960161280E+33f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(10, 10), 8.093278209760000E+12f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(10, -10), 8.093278209760000E+12f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(3, -10), -7.880000000000000E+3f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(3, -1000), -7.999988000000000E+9f, 100 * eps, __LINE__);
   check_close_fraction_f(hermitef(3, -1000000), -7.999999999988000E+18f, 100 * eps, __LINE__);

   check_close_f(riemann_zetaf(0.125), -0.63277562349869525529352526763564627152686379131122f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(1023 / 1024.0f), -1023.4228554489429786541032870895167448906103303056f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(1025 / 1024.0f), 1024.5772867695045940578681624248887776501597556226f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(0.5f), -1.46035450880958681288949915251529801246722933101258149054289f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(1.125f), 8.5862412945105752999607544082693023591996301183069f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(2), 1.6449340668482264364724151666460251892189499012068f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(3.5f), 1.1267338673170566464278124918549842722219969574036f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(4), 1.08232323371113819151600369654116790277475095191872690768298f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(4 + 1 / 1024.0f), 1.08225596856391369799036835439238249195298434901488518878804f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(4.5f), 1.05470751076145426402296728896028011727249383295625173068468f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(6.5f), 1.01200589988852479610078491680478352908773213619144808841031f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(7.5f), 1.00582672753652280770224164440459408011782510096320822989663f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(8.125f), 1.0037305205308161603183307711439385250181080293472f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(16.125f), 1.0000140128224754088474783648500235958510030511915f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(0), -0.5f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-0.125f), -0.39906966894504503550986928301421235400280637468895f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-1), -0.083333333333333333333333333333333333333333333333333f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-2), 0, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-2.5f), 0.0085169287778503305423585670283444869362759902200745f, eps * 5000 * 3, __LINE__);
   check_close_f(riemann_zetaf(-3), 0.0083333333333333333333333333333333333333333333333333f, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-4), 0, eps * 5000, __LINE__);
   check_close_f(riemann_zetaf(-20), 0, eps * 5000 * 100, __LINE__);
   check_close_f(riemann_zetaf(-21), -281.46014492753623188405797101449275362318840579710f, eps * 5000 * 100, __LINE__);
   check_close_f(riemann_zetaf(-30.125f), 2.2762941726834511267740045451463455513839970804578e7f, eps * 5000 * 100, __LINE__);

   check_close_f(sph_besself(0, 0.1433600485324859619140625e-1f), 0.9999657468461303487880990241993035937654e0f,  eps * 5000 * 100, __LINE__);
   check_close_f(sph_besself(0, 0.1760916970670223236083984375e-1f), 0.9999483203249623334100130061926184665364e0f,  eps * 5000 * 100, __LINE__);
   check_close_f(sph_besself(2, 0.1433600485324859619140625e-1f), 0.1370120120703995134662099191103188366059e-4f,  eps * 5000 * 100, __LINE__);
   check_close_f(sph_besself(2, 0.1760916970670223236083984375e-1f), 0.2067173265753174063228459655801741280461e-4f,  eps * 5000 * 100, __LINE__);
   check_close_f(sph_besself(7, 0.1252804412841796875e3f), 0.7887555711993028736906736576314283291289e-2f,  eps * 5000 * 100, __LINE__);
   check_close_f(sph_besself(7, 0.25554705810546875e3f), -0.1463292767579579943284849187188066532514e-2f,  eps * 5000 * 100, __LINE__);

   check_close_f(sph_neumannf(0, 0.408089816570281982421875e0f), -0.2249212131304610409189209411089291558038e1f, eps * 5000 * 100, __LINE__);
   check_close_f(sph_neumannf(0, 0.6540834903717041015625e0f), -0.1213309779166084571756446746977955970241e1f,   eps * 5000 * 100, __LINE__);
   check_close_f(sph_neumannf(2, 0.408089816570281982421875e0f), -0.4541702641837159203058389758895634766256e2f,   eps * 5000 * 100, __LINE__);
   check_close_f(sph_neumannf(2, 0.6540834903717041015625e0f), -0.1156112621471167110574129561700037138981e2f,   eps * 5000 * 100, __LINE__);
   check_close_f(sph_neumannf(10, 0.1097540378570556640625e1f), -0.2427889658115064857278886600528596240123e9f,   eps * 5000 * 100, __LINE__);
   check_close_f(sph_neumannf(10, 0.30944411754608154296875e1f), -0.3394649246350136450439882104151313759251e4f,   eps * 5000 * 100, __LINE__);

   check_close_fraction_f(sph_legendref(3, 2, 0.5f), 0.2061460599687871330692286791802688341213f, eps * 5000, __LINE__);
   check_close_fraction_f(sph_legendref(40, 15, 0.75f), -0.406036847302819452666908966769096223205057182668333862900509f, eps * 5000, __LINE__);

#endif
}

void test_values(const char* name)
{
#ifndef TEST_LD
   //
   // First the C99 math functions:
   //
   double eps = DBL_EPSILON;
   double sv;
   check_close(acosh(cosh(0.5)), 0.5, 5000 * eps, __LINE__);
   check_close(asinh(sinh(0.5)), 0.5, 5000 * eps, __LINE__);
   check_close(atanh(tanh(0.5)), 0.5, 5000 * eps, __LINE__);

   check_close(cbrt(1.5 * 1.5 * 1.5), 1.5, 5000 * eps, __LINE__);

   check(copysign(1.0, 1.0) == 1.0, __LINE__);
   check(copysign(1.0, -1.0) == -1.0, __LINE__);
   check(copysign(-1.0, 1.0) == 1.0, __LINE__);
   check(copysign(-1.0, -1.0) == -1.0, __LINE__);

   check_close(erfc(0.125), 0.85968379519866618260697055347837660181302041685015, eps * 1000, __LINE__);
   check_close(erfc(0.5), 0.47950012218695346231725334610803547126354842424204, eps * 1000, __LINE__);
   check_close(erfc(1), 0.15729920705028513065877936491739074070393300203370, eps * 1000, __LINE__);
   check_close(erfc(5), 1.5374597944280348501883434853833788901180503147234e-12, eps * 1000, __LINE__);
   check_close(erfc(-0.125), 1.1403162048013338173930294465216233981869795831498, eps * 1000, __LINE__);
   check_close(erfc(-0.5), 1.5204998778130465376827466538919645287364515757580, eps * 1000, __LINE__);
   check_close(erfc(0), 1, eps * 1000, __LINE__);

   check_close(erf(0.125), 0.14031620480133381739302944652162339818697958314985, eps * 1000, __LINE__);
   check_close(erf(0.5), 0.52049987781304653768274665389196452873645157575796, eps * 1000, __LINE__);
   check_close(erf(1), 0.84270079294971486934122063508260925929606699796630, eps * 1000, __LINE__);
   check_close(erf(5), 0.9999999999984625402055719651498116565146166211099, eps * 1000, __LINE__);
   check_close(erf(-0.125), -0.14031620480133381739302944652162339818697958314985, eps * 1000, __LINE__);
   check_close(erf(-0.5), -0.52049987781304653768274665389196452873645157575796, eps * 1000, __LINE__);
   check_close(erf(0), 0, eps * 1000, __LINE__);

   check_close(log1p(0.582029759883880615234375e0), 0.4587086807259736626531803258754840111707e0, eps * 1000, __LINE__);
   check_close(expm1(0.582029759883880615234375e0), 0.7896673415707786528734865994546559029663e0, eps * 1000, __LINE__);
   check_close(log1p(-0.2047410048544406890869140625e-1), -0.2068660038044094868521052319477265955827e-1, eps * 1000, __LINE__);
   check_close(expm1(-0.2047410048544406890869140625e-1), -0.2026592921724753704129022027337835687888e-1, eps * 1000, __LINE__);

   check_close(fmax(0.1, -0.1), 0.1, 0, __LINE__);
   check_close(fmin(0.1, -0.1), -0.1, 0, __LINE__);

   check_close(hypot(1.0, 3.0), sqrt(10.0), eps * 500, __LINE__);

   check_close(lgamma(3.5), 1.2009736023470742248160218814507129957702389154682, 5000 * eps, __LINE__);
   check_close(lgamma(0.125), 2.0194183575537963453202905211670995899482809521344, 5000 * eps, __LINE__);
   check_close(lgamma(-0.125), 2.1653002489051702517540619481440174064962195287626, 5000 * eps, __LINE__);
   check_close(lgamma(-3.125), 0.1543111276840418242676072830970532952413339012367, 5000 * eps, __LINE__);
   check_close(lgamma(-53249.0/1024), -149.43323093420259741100038126078721302600128285894, 5000 * eps, __LINE__);

   check_close(tgamma(3.5), 3.3233509704478425511840640312646472177454052302295, 5000 * eps, __LINE__);
   check_close(tgamma(0.125), 7.5339415987976119046992298412151336246104195881491, 5000 * eps, __LINE__);
   check_close(tgamma(-0.125), -8.7172188593831756100190140408231437691829605421405, 5000 * eps, __LINE__);
   check_close(tgamma(-3.125), 1.1668538708507675587790157356605097019141636072094, 5000 * eps, __LINE__);

#ifdef BOOST_HAS_LONG_LONG
   check(llround(2.5) == 3L, __LINE__);
   check(llround(2.25) == 2L, __LINE__);
#endif
   check(lround(2.5) == 3.0, __LINE__);
   check(lround(2.25) == 2.0, __LINE__);
   check(round(2.5) == 3.0, __LINE__);
   check(round(2.25) == 2.0, __LINE__);

   check(nextafter(1.0, 2.0) > 1.0, __LINE__);
   check(nextafter(1.0, -2.0) < 1.0, __LINE__);
   check(nextafter(nextafter(1.0, 2.0), -2.0) == 1.0, __LINE__);
   check(nextafter(nextafter(1.0, -2.0), 2.0) == 1.0, __LINE__);
   check(nextafter(1.0, 2.0) > 1.0, __LINE__);
   check(nextafter(1.0, -2.0) < 1.0, __LINE__);
   check(nextafter(nextafter(1.0, 2.0), -2.0) == 1.0, __LINE__);
   check(nextafter(nextafter(1.0, -2.0), 2.0) == 1.0, __LINE__);

   check(trunc(2.5) == 2.0, __LINE__);
   check(trunc(2.25) == 2.0, __LINE__);

   //
   // Now for the TR1 math functions:
   //
   check_close_fraction(assoc_laguerre(4, 5, 0.5), 88.31510416666666666666666666666666666667, eps * 100, __LINE__);
   check_close_fraction(assoc_laguerre(10, 0, 2.5), -0.8802526766660982969576719576719576719577, eps * 100, __LINE__);
   check_close_fraction(assoc_laguerre(10, 1, 4.5), 1.564311458042689732142857142857142857143, eps * 100, __LINE__);
   check_close_fraction(assoc_laguerre(10, 6, 8.5), 20.51596541066649098875661375661375661376, eps * 100, __LINE__);
   check_close_fraction(assoc_laguerre(10, 12, 12.5), -199.5560968456234671241181657848324514991, eps * 100, __LINE__);
   check_close_fraction(assoc_laguerre(50, 40, 12.5), -4.996769495006119488583146995907246595400e16, eps * 100, __LINE__);

   check_close_fraction(laguerre(1, 0.5), 0.5, eps * 100, __LINE__);
   check_close_fraction(laguerre(4, 0.5), -0.3307291666666666666666666666666666666667, eps * 100, __LINE__);
   check_close_fraction(laguerre(7, 0.5), -0.5183392237103174603174603174603174603175, eps * 100, __LINE__);
   check_close_fraction(laguerre(20, 0.5), 0.3120174870800154148915399248893113634676, eps * 100, __LINE__);
   check_close_fraction(laguerre(50, 0.5), -0.3181388060269979064951118308575628226834, eps * 100, __LINE__);

   check_close_fraction(laguerre(1, -0.5), 1.5, eps * 100, __LINE__);
   check_close_fraction(laguerre(4, -0.5), 3.835937500000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(laguerre(7, -0.5), 7.950934709821428571428571428571428571429, eps * 100, __LINE__);
   check_close_fraction(laguerre(20, -0.5), 76.12915699869631476833699787070874048223, eps * 100, __LINE__);
   check_close_fraction(laguerre(50, -0.5), 2307.428631277506570629232863491518399720, eps * 100, __LINE__);

   check_close_fraction(laguerre(1, 4.5), -3.500000000000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(laguerre(4, 4.5), 0.08593750000000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(laguerre(7, 4.5), -1.036928013392857142857142857142857142857, eps * 100, __LINE__);
   check_close_fraction(laguerre(20, 4.5), 1.437239150257817378525582974722170737587, eps * 100, __LINE__);
   check_close_fraction(laguerre(50, 4.5), -0.7795068145562651416494321484050019245248, eps * 100, __LINE__);

   check_close_fraction(assoc_legendre(4, 2, 0.5), 4.218750000000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(assoc_legendre(7, 5, 0.5), 5696.789530152175143607977274672800795328, eps * 100, __LINE__);
   check_close_fraction(assoc_legendre(4, 2, -0.5), 4.218750000000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(assoc_legendre(7, 5, -0.5), 5696.789530152175143607977274672800795328, eps * 100, __LINE__);

   check_close_fraction(legendre(1, 0.5), 0.5, eps * 100, __LINE__);
   check_close_fraction(legendre(4, 0.5), -0.2890625000000000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(legendre(7, 0.5), 0.2231445312500000000000000000000000000000, eps * 100, __LINE__);
   check_close_fraction(legendre(40, 0.5), -0.09542943523261546936538467572384923220258, eps * 100, __LINE__);

   sv = eps / 1024;
   check_close(beta(1, 1), 1, eps * 20 * 100, __LINE__);
   check_close(beta(1, 4), 0.25, eps * 20 * 100, __LINE__);
   check_close(beta(4, 1), 0.25, eps * 20 * 100, __LINE__);
   check_close(beta(sv, 4), 1/sv, eps * 20 * 100, __LINE__);
   check_close(beta(4, sv), 1/sv, eps * 20 * 100, __LINE__);
   check_close(beta(4, 20), 0.00002823263692828910220214568040654997176736, eps * 20 * 100, __LINE__);
   check_close(beta(0.0125, 0.000023), 43558.24045647538375006349016083320744662, eps * 20 * 100, __LINE__);

   check_close(comp_ellint_1(0), 1.5707963267948966192313216916397514420985846996876, eps * 5000, __LINE__);
   check_close(comp_ellint_1(0.125), 1.5769867712158131421244030532288080803822271060839, eps * 5000, __LINE__);
   check_close(comp_ellint_1(0.25), 1.5962422221317835101489690714979498795055744578951, eps * 5000, __LINE__);
   check_close(comp_ellint_1(300/1024.0), 1.6062331054696636704261124078746600894998873503208, eps * 5000, __LINE__);
   check_close(comp_ellint_1(400/1024.0), 1.6364782007562008756208066125715722889067992997614, eps * 5000, __LINE__);
   check_close(comp_ellint_1(-0.5), 1.6857503548125960428712036577990769895008008941411, eps * 5000, __LINE__);
   check_close(comp_ellint_1(-0.75), 1.9109897807518291965531482187613425592531451316788, eps * 5000, __LINE__);

   check_close(comp_ellint_2(-1), 1.0, eps * 5000, __LINE__);
   check_close(comp_ellint_2(0), 1.5707963267948966192313216916397514420985846996876, eps * 5000, __LINE__);
   check_close(comp_ellint_2(100 / 1024.0), 1.5670445330545086723323795143598956428788609133377, eps * 5000, __LINE__);
   check_close(comp_ellint_2(200 / 1024.0), 1.5557071588766556854463404816624361127847775545087, eps * 5000, __LINE__);
   check_close(comp_ellint_2(300 / 1024.0), 1.5365278991162754883035625322482669608948678755743, eps * 5000, __LINE__);
   check_close(comp_ellint_2(400 / 1024.0), 1.5090417763083482272165682786143770446401437564021, eps * 5000, __LINE__);
   check_close(comp_ellint_2(-0.5), 1.4674622093394271554597952669909161360253617523272, eps * 5000, __LINE__);
   check_close(comp_ellint_2(-600 / 1024.0), 1.4257538571071297192428217218834579920545946473778, eps * 5000, __LINE__);
   check_close(comp_ellint_2(-800 / 1024.0), 1.2927868476159125056958680222998765985004489572909, eps * 5000, __LINE__);
   check_close(comp_ellint_2(-900 / 1024.0), 1.1966864890248739524112920627353824133420353430982, eps * 5000, __LINE__);

   check_close(comp_ellint_3(0.2, 0), 1.586867847454166237308008033828114192951, eps * 5000, __LINE__);
   check_close(comp_ellint_3(0.4, 0), 1.639999865864511206865258329748601457626, eps * 5000, __LINE__);
   check_close(comp_ellint_3(0.0, 0), 1.57079632679489661923132169163975144209858469968755291048747, eps * 5000, __LINE__);
   check_close(comp_ellint_3(0.0, 0.5), 2.221441469079183123507940495030346849307, eps * 5000, __LINE__);
   check_close(comp_ellint_3(0.3, -4), 0.712708870925620061597924858162260293305195624270730660081949, eps * 5000, __LINE__);
   check_close(comp_ellint_3(-0.5, -1e+05), 0.00496944596485066055800109163256108604615568144080386919012831, eps * 5000, __LINE__);
   check_close(comp_ellint_3(-0.75, -1e+10), 0.0000157080225184890546939710019277357161497407143903832703317801, eps * 5000, __LINE__);
   check_close(comp_ellint_3(-0.875, 1 / 1024.0), 2.18674503176462374414944618968850352696579451638002110619287, eps * 5000, __LINE__);
   check_close(comp_ellint_3(-0.875, 1023/1024.0), 101.045289804941384100960063898569538919135722087486350366997, eps * 5000, __LINE__);

   check_close(cyl_bessel_i(2.25, 1/(1024.0*1024.0)), 2.34379212133481347189068464680335815256364262507955635911656e-15, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(5.5, 3.125), 0.0583514045989371500460946536220735787163510569634133670181210, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(-5 + 1.0/1024.0, 2.125), 0.0267920938009571023702933210070984416052633027166975342895062, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(-5.5, 10), 597.577606961369169607937419869926705730305175364662688426534, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(-10486074.0/(1024.0*1024), 1/1024.0), 1.41474005665181350367684623930576333542989766867888186478185e35, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(-10486074.0/(1024.0*1024), 50), 1.07153277202900671531087024688681954238311679648319534644743e20, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(144794.0/1024.0, 100), 2066.27694757392660413922181531984160871678224178890247540320, eps * 5000, __LINE__);
   check_close(cyl_bessel_i(-144794.0/1024.0, 100), 2066.27694672763190927440969155740243346136463461655104698748, eps * 5000, __LINE__);

   check_close(cyl_bessel_j(2457/1024.0, 1/1024.0), 3.80739920118603335646474073457326714709615200130620574875292e-9, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(5.5, 3217.0/1024), 0.0281933076257506091621579544064767140470089107926550720453038, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-5.5, 3217.0/1024), -2.55820064470647911823175836997490971806135336759164272675969, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-5.5, 1e+04), 2.449843111985605522111159013846599118397e-03, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(5.5, 1e+04), 0.00759343502722670361395585198154817047185480147294665270646578, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(5.5, 1e+06), -0.000747424248595630177396350688505919533097973148718960064663632, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(5.125, 1e+06), -0.000776600124835704280633640911329691642748783663198207360238214, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(5.875, 1e+06), -0.000466322721115193071631008581529503095819705088484386434589780, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(0.5, 101), 0.0358874487875643822020496677692429287863419555699447066226409, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-5.5, 1e+04), 0.00244984311198560552211115901384659911839737686676766460822577, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-5.5, 1e+06), 0.000279243200433579511095229508894156656558211060453622750659554, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-0.5, 101), 0.0708184798097594268482290389188138201440114881159344944791454, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-10486074 / (1024*1024.0), 1/1024.0), 1.41474013160494695750009004222225969090304185981836460288562e35, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-10486074 / (1024*1024.0), 15), -0.0902239288885423309568944543848111461724911781719692852541489, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(10486074 / (1024*1024.0), 1e+02), -0.0547064914615137807616774867984047583596945624129838091326863, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(10486074 / (1024*1024.0), 2e+04), -0.00556783614400875611650958980796060611309029233226596737701688, eps * 5000, __LINE__);
   check_close(cyl_bessel_j(-10486074 / (1024*1024.0), 1e+02), -0.0547613660316806551338637153942604550779513947674222863858713, eps * 5000, __LINE__);

   check_close(cyl_bessel_k(0.5, 0.875), 0.558532231646608646115729767013630967055657943463362504577189, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(0.5, 1.125), 0.383621010650189547146769320487006220295290256657827220786527, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(2.25, ldexp(1.0, -30)), 5.62397392719283271332307799146649700147907612095185712015604e20, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(5.5, 3217/1024.0), 1.30623288775012596319554857587765179889689223531159532808379, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(-5.5, 10), 0.0000733045300798502164644836879577484533096239574909573072142667, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(-5.5, 100), 5.41274555306792267322084448693957747924412508020839543293369e-45, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(10240/1024.0, 1/1024.0), 2.35522579263922076203415803966825431039900000000993410734978e38, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(10240/1024.0, 10), 0.00161425530039067002345725193091329085443750382929208307802221, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(144793/1024.0, 100), 1.39565245860302528069481472855619216759142225046370312329416e-6, eps * 5000, __LINE__);
   check_close(cyl_bessel_k(144793/1024.0, 200), 9.11950412043225432171915100042647230802198254567007382956336e-68, eps * 5000, __LINE__);

   check_close(cyl_neumann(0.5, 1 / (1024.0*1024)), -817.033790261762580469303126467917092806755460418223776544122, eps * 5000, __LINE__);
   check_close(cyl_neumann(5.5, 3.125), -2.61489440328417468776474188539366752698192046890955453259866, eps * 5000, __LINE__);
   check_close(cyl_neumann(-5.5, 3.125), -0.0274994493896489729948109971802244976377957234563871795364056, eps * 5000, __LINE__);
   check_close(cyl_neumann(-5.5, 1e+04), -0.00759343502722670361395585198154817047185480147294665270646578, eps * 5000, __LINE__);
   check_close(cyl_neumann(-10486074 / (1024*1024.0), 1/1024.0), -1.50382374389531766117868938966858995093408410498915220070230e38, eps * 5000, __LINE__);
   check_close(cyl_neumann(-10486074 / (1024*1024.0), 1e+02), 0.0583041891319026009955779707640455341990844522293730214223545, eps * 5000, __LINE__);
   check_close(cyl_neumann(141.75, 1e+02), -5.38829231428696507293191118661269920130838607482708483122068e9, eps * 5000, __LINE__);
   check_close(cyl_neumann(141.75, 2e+04), -0.00376577888677186194728129112270988602876597726657372330194186, eps * 5000, __LINE__);
   check_close(cyl_neumann(-141.75, 1e+02), -3.81009803444766877495905954105669819951653361036342457919021e9, eps * 5000, __LINE__);

   check_close(ellint_1(0, 0), 0, eps * 5000, __LINE__);
   check_close(ellint_1(0, -10), -10, eps * 5000, __LINE__);
   check_close(ellint_1(-1, -1), -1.2261911708835170708130609674719067527242483502207, eps * 5000, __LINE__);
   check_close(ellint_1(0.875, -4), -5.3190556182262405182189463092940736859067548232647, eps * 5000, __LINE__);
   check_close(ellint_1(-0.625, 8), 9.0419973860310100524448893214394562615252527557062, eps * 5000, __LINE__);
   check_close(ellint_1(0.875, 1e-05), 0.000010000000000127604166668510945638036143355898993088, eps * 5000, __LINE__);
   check_close(ellint_1(10/1024.0, 1e+05), 100002.38431454899771096037307519328741455615271038, eps * 5000, __LINE__);
   check_close(ellint_1(1, 1e-20), 1.0000000000000000000000000000000000000000166666667e-20, eps * 5000, __LINE__);
   check_close(ellint_1(1e-20, 1e-20), 1.000000000000000e-20, eps * 5000, __LINE__);
   check_close(ellint_1(400/1024.0, 1e+20), 1.0418143796499216839719289963154558027005142709763e20, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, 2), 2.1765877052210673672479877957388515321497888026770, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, 4), 4.2543274975235836861894752787874633017836785640477, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, 6), 6.4588766202317746302999080620490579800463614807916, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, 10), 10.697409951222544858346795279378531495869386960090, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, -2), -2.1765877052210673672479877957388515321497888026770, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, -4), -4.2543274975235836861894752787874633017836785640477, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, -6), -6.4588766202317746302999080620490579800463614807916, eps * 5000, __LINE__);
   check_close(ellint_1(0.5, -10), -10.697409951222544858346795279378531495869386960090, eps * 5000, __LINE__);

   check_close(ellint_2(0, 0), 0, eps * 5000, __LINE__);
   check_close(ellint_2(0, -10), -10, eps * 5000, __LINE__);
   check_close(ellint_2(-1, -1), -0.84147098480789650665250232163029899962256306079837, eps * 5000, __LINE__);
   check_close(ellint_2(900 / 1024.0, -4), -3.1756145986492562317862928524528520686391383168377, eps * 5000, __LINE__);
   check_close(ellint_2(-600 / 1024.0, 8), 7.2473147180505693037677015377802777959345489333465, eps * 5000, __LINE__);
   check_close(ellint_2(800 / 1024.0, 1e-05), 9.999999999898274739584436515967055859383969942432E-6, eps * 5000, __LINE__);
   check_close(ellint_2(100 / 1024.0, 1e+05), 99761.153306972066658135668386691227343323331995888, eps * 5000, __LINE__);
   check_close(ellint_2(-0.5, 1e+10), 9.3421545766487137036576748555295222252286528414669e9, eps * 5000, __LINE__);
   check_close(ellint_2(400 / 1024.0, ldexp(1, 66)), 7.0886102721911705466476846969992069994308167515242e19, eps * 5000, __LINE__);

   check_close(ellint_3(0, 1, -1), -1.557407724654902230506974807458360173087, eps * 5000, __LINE__);
   check_close(ellint_3(0.4, 0, -4), -4.153623371196831087495427530365430979011, eps * 5000, __LINE__);
   check_close(ellint_3(-0.6, 0, 8), 8.935930619078575123490612395578518914416, eps * 5000, __LINE__);
   check_close(ellint_3(0.25, 0, 0.5), 0.501246705365439492445236118603525029757890291780157969500480, eps * 5000, __LINE__);
   check_close(ellint_3(0, 0, 0.5), 0.5, eps * 5000, __LINE__);
   check_close(ellint_3(0, -2, 0.5), 0.437501067017546278595664813509803743009132067629603474488486, eps * 5000, __LINE__);
   check_close(ellint_3(0, 0.25, 0.5), 0.510269830229213412212501938035914557628394166585442994564135, eps * 5000, __LINE__);
   check_close(ellint_3(0, 0.75, 0.5), 0.533293253875952645421201146925578536430596894471541312806165, eps * 5000, __LINE__);
   check_close(ellint_3(0, 0.75, 0.75), 0.871827580412760575085768367421866079353646112288567703061975, eps * 5000, __LINE__);
   check_close(ellint_3(0, 1, 0.25), 0.255341921221036266504482236490473678204201638800822621740476, eps * 5000, __LINE__);
   check_close(ellint_3(0, 2, 0.25), 0.261119051639220165094943572468224137699644963125853641716219, eps * 5000, __LINE__);
   check_close(ellint_3(0, 1023/1024.0, 1.5), 13.2821612239764190363647953338544569682942329604483733197131, eps * 5000, __LINE__);
   check_close(ellint_3(0.5, 0.5, -1), -1.228014414316220642611298946293865487807, eps * 5000, __LINE__);
   check_close(ellint_3(0.5, 0.5, 1e+10), 1.536591003599172091573590441336982730551e+10, eps * 5000, __LINE__);
   check_close(ellint_3(0.75, -1e+05, 10), 0.0347926099493147087821620459290460547131012904008557007934290, eps * 5000, __LINE__);
   check_close(ellint_3(0.875, -1e+10, 10), 0.000109956202759561502329123384755016959364346382187364656768212, eps * 5000, __LINE__);
   check_close(ellint_3(0.875, -1e+10, 1e+20), 1.00000626665567332602765201107198822183913978895904937646809e15, eps * 5000, __LINE__);
   check_close(ellint_3(0.875, -1e+10, 1608/1024.0), 0.0000157080616044072676127333183571107873332593142625043567690379, eps * 5000, __LINE__);
   check_close(ellint_3(0.875, 1-1 / 1024.0, 1e+20), 6.43274293944380717581167058274600202023334985100499739678963e21, eps * 5000, __LINE__);
   check_close(ellint_3(0.25, 50, 0.1), 0.124573770342749525407523258569507331686458866564082916835900, eps * 5000, __LINE__);
   check_close(ellint_3(0.25, 1.125, 1), 1.77299767784815770192352979665283069318388205110727241629752, eps * 5000, __LINE__);

   check_close(expint(1/1024.0), -6.35327933972759151358547423727042905862963067106751711596065, eps * 5000, __LINE__);
   check_close(expint(0.125), -1.37320852494298333781545045921206470808223543321810480716122, eps * 5000, __LINE__);
   check_close(expint(0.5), 0.454219904863173579920523812662802365281405554352642045162818, eps * 5000, __LINE__);
   check_close(expint(1), 1.89511781635593675546652093433163426901706058173270759164623, eps * 5000, __LINE__);
   check_close(expint(50.5), 1.72763195602911805201155668940185673806099654090456049881069e20, eps * 5000, __LINE__);
   check_close(expint(-1/1024.0), -6.35523246483107180261445551935803221293763008553775821607264, eps * 5000, __LINE__);
   check_close(expint(-0.125), -1.62342564058416879145630692462440887363310605737209536579267, eps * 5000, __LINE__);
   check_close(expint(-0.5), -0.559773594776160811746795939315085235226846890316353515248293, eps * 5000, __LINE__);
   check_close(expint(-1), -0.219383934395520273677163775460121649031047293406908207577979, eps * 5000, __LINE__);
   check_close(expint(-50.5), -2.27237132932219350440719707268817831250090574830769670186618e-24, eps * 5000, __LINE__);

   check_close_fraction(hermite(0, 1), 1.L, 100 * eps, __LINE__);
   check_close_fraction(hermite(1, 1), 2.L, 100 * eps, __LINE__);
   check_close_fraction(hermite(1, 2), 4.L, 100 * eps, __LINE__);
   check_close_fraction(hermite(1, 10), 20, 100 * eps, __LINE__);
   check_close_fraction(hermite(1, 100), 200, 100 * eps, __LINE__);
   check_close_fraction(hermite(1, 1e6), 2e6, 100 * eps, __LINE__);
   check_close_fraction(hermite(10, 30), 5.896624628001300E+17, 100 * eps, __LINE__);
   check_close_fraction(hermite(10, 1000), 1.023976960161280E+33, 100 * eps, __LINE__);
   check_close_fraction(hermite(10, 10), 8.093278209760000E+12, 100 * eps, __LINE__);
   check_close_fraction(hermite(10, -10), 8.093278209760000E+12, 100 * eps, __LINE__);
   check_close_fraction(hermite(3, -10), -7.880000000000000E+3, 100 * eps, __LINE__);
   check_close_fraction(hermite(3, -1000), -7.999988000000000E+9, 100 * eps, __LINE__);
   check_close_fraction(hermite(3, -1000000), -7.999999999988000E+18, 100 * eps, __LINE__);

   check_close(riemann_zeta(0.125), -0.63277562349869525529352526763564627152686379131122, eps * 5000, __LINE__);
   check_close(riemann_zeta(1023 / 1024.0), -1023.4228554489429786541032870895167448906103303056, eps * 5000, __LINE__);
   check_close(riemann_zeta(1025 / 1024.0), 1024.5772867695045940578681624248887776501597556226, eps * 5000, __LINE__);
   check_close(riemann_zeta(0.5), -1.46035450880958681288949915251529801246722933101258149054289, eps * 5000, __LINE__);
   check_close(riemann_zeta(1.125), 8.5862412945105752999607544082693023591996301183069, eps * 5000, __LINE__);
   check_close(riemann_zeta(2), 1.6449340668482264364724151666460251892189499012068, eps * 5000, __LINE__);
   check_close(riemann_zeta(3.5), 1.1267338673170566464278124918549842722219969574036, eps * 5000, __LINE__);
   check_close(riemann_zeta(4), 1.08232323371113819151600369654116790277475095191872690768298, eps * 5000, __LINE__);
   check_close(riemann_zeta(4 + 1 / 1024.0), 1.08225596856391369799036835439238249195298434901488518878804, eps * 5000, __LINE__);
   check_close(riemann_zeta(4.5), 1.05470751076145426402296728896028011727249383295625173068468, eps * 5000, __LINE__);
   check_close(riemann_zeta(6.5), 1.01200589988852479610078491680478352908773213619144808841031, eps * 5000, __LINE__);
   check_close(riemann_zeta(7.5), 1.00582672753652280770224164440459408011782510096320822989663, eps * 5000, __LINE__);
   check_close(riemann_zeta(8.125), 1.0037305205308161603183307711439385250181080293472, eps * 5000, __LINE__);
   check_close(riemann_zeta(16.125), 1.0000140128224754088474783648500235958510030511915, eps * 5000, __LINE__);
   check_close(riemann_zeta(0), -0.5, eps * 5000, __LINE__);
   check_close(riemann_zeta(-0.125), -0.39906966894504503550986928301421235400280637468895, eps * 5000, __LINE__);
   check_close(riemann_zeta(-1), -0.083333333333333333333333333333333333333333333333333, eps * 5000, __LINE__);
   check_close(riemann_zeta(-2), 0, eps * 5000, __LINE__);
   check_close(riemann_zeta(-2.5), 0.0085169287778503305423585670283444869362759902200745, eps * 5000 * 3, __LINE__);
   check_close(riemann_zeta(-3), 0.0083333333333333333333333333333333333333333333333333, eps * 5000, __LINE__);
   check_close(riemann_zeta(-4), 0, eps * 5000, __LINE__);
   check_close(riemann_zeta(-20), 0, eps * 5000 * 100, __LINE__);
   check_close(riemann_zeta(-21), -281.46014492753623188405797101449275362318840579710, eps * 5000 * 100, __LINE__);
   check_close(riemann_zeta(-30.125), 2.2762941726834511267740045451463455513839970804578e7, eps * 5000 * 100, __LINE__);

   check_close(sph_bessel(0, 0.1433600485324859619140625e-1), 0.9999657468461303487880990241993035937654e0,  eps * 5000 * 100, __LINE__);
   check_close(sph_bessel(0, 0.1760916970670223236083984375e-1), 0.9999483203249623334100130061926184665364e0,  eps * 5000 * 100, __LINE__);
   check_close(sph_bessel(2, 0.1433600485324859619140625e-1), 0.1370120120703995134662099191103188366059e-4,  eps * 5000 * 100, __LINE__);
   check_close(sph_bessel(2, 0.1760916970670223236083984375e-1), 0.2067173265753174063228459655801741280461e-4,  eps * 5000 * 100, __LINE__);
   check_close(sph_bessel(7, 0.1252804412841796875e3), 0.7887555711993028736906736576314283291289e-2,  eps * 5000 * 100, __LINE__);
   check_close(sph_bessel(7, 0.25554705810546875e3), -0.1463292767579579943284849187188066532514e-2,  eps * 5000 * 100, __LINE__);

   check_close(sph_neumann(0, 0.408089816570281982421875e0), -0.2249212131304610409189209411089291558038e1, eps * 5000 * 100, __LINE__);
   check_close(sph_neumann(0, 0.6540834903717041015625e0), -0.1213309779166084571756446746977955970241e1,   eps * 5000 * 100, __LINE__);
   check_close(sph_neumann(2, 0.408089816570281982421875e0), -0.4541702641837159203058389758895634766256e2,   eps * 5000 * 100, __LINE__);
   check_close(sph_neumann(2, 0.6540834903717041015625e0), -0.1156112621471167110574129561700037138981e2,   eps * 5000 * 100, __LINE__);
   check_close(sph_neumann(10, 0.1097540378570556640625e1), -0.2427889658115064857278886600528596240123e9,   eps * 5000 * 100, __LINE__);
   check_close(sph_neumann(10, 0.30944411754608154296875e1), -0.3394649246350136450439882104151313759251e4,   eps * 5000 * 100, __LINE__);

   check_close_fraction(sph_legendre(3, 2, 0.5), 0.2061460599687871330692286791802688341213, eps * 5000, __LINE__);
   check_close_fraction(sph_legendre(40, 15, 0.75), -0.406036847302819452666908966769096223205057182668333862900509, eps * 5000, __LINE__);
#endif
}

void test_valuesl(const char* name)
{
#ifdef TEST_LD
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   //
   // First the C99 math functions:
   //
   long double eps = LDBL_EPSILON;
   long double sv;
   check_close_l(acoshl(coshl(0.5L)), 0.5L, 5000 * eps, __LINE__);
   check_close_l(asinhl(sinhl(0.5L)), 0.5L, 5000 * eps, __LINE__);
   check_close_l(atanhl(tanhl(0.5L)), 0.5L, 5000 * eps, __LINE__);

   check_close_l(cbrtl(1.5L * 1.5L * 1.5L), 1.5L, 5000 * eps, __LINE__);

   check(copysignl(1.0L, 1.0L) == 1.0L, __LINE__);
   check(copysignl(1.0L, -1.0L) == -1.0L, __LINE__);
   check(copysignl(-1.0L, 1.0L) == 1.0L, __LINE__);
   check(copysignl(-1.0L, -1.0L) == -1.0L, __LINE__);

   check_close_l(erfcl(0.125), 0.85968379519866618260697055347837660181302041685015L, eps * 1000, __LINE__);
   check_close_l(erfcl(0.5), 0.47950012218695346231725334610803547126354842424204L, eps * 1000, __LINE__);
   check_close_l(erfcl(1), 0.15729920705028513065877936491739074070393300203370L, eps * 1000, __LINE__);
   check_close_l(erfcl(5), 1.5374597944280348501883434853833788901180503147234e-12L, eps * 1000, __LINE__);
   check_close_l(erfcl(-0.125), 1.1403162048013338173930294465216233981869795831498L, eps * 1000, __LINE__);
   check_close_l(erfcl(-0.5), 1.5204998778130465376827466538919645287364515757580L, eps * 1000, __LINE__);
   check_close_l(erfcl(0), 1, eps * 1000, __LINE__);

   check_close_l(erfl(0.125), 0.14031620480133381739302944652162339818697958314985L, eps * 1000, __LINE__);
   check_close_l(erfl(0.5), 0.52049987781304653768274665389196452873645157575796L, eps * 1000, __LINE__);
   check_close_l(erfl(1), 0.84270079294971486934122063508260925929606699796630L, eps * 1000, __LINE__);
   check_close_l(erfl(5), 0.9999999999984625402055719651498116565146166211099L, eps * 1000, __LINE__);
   check_close_l(erfl(-0.125), -0.14031620480133381739302944652162339818697958314985L, eps * 1000, __LINE__);
   check_close_l(erfl(-0.5), -0.52049987781304653768274665389196452873645157575796L, eps * 1000, __LINE__);
   check_close_l(erfl(0), 0, eps * 1000, __LINE__);

   check_close_l(log1pl(0.582029759883880615234375e0), 0.4587086807259736626531803258754840111707e0L, eps * 1000, __LINE__);
   check_close_l(expm1l(0.582029759883880615234375e0), 0.7896673415707786528734865994546559029663e0L, eps * 1000, __LINE__);
   check_close_l(log1pl(-0.2047410048544406890869140625e-1), -0.2068660038044094868521052319477265955827e-1L, eps * 1000, __LINE__);
   check_close_l(expm1l(-0.2047410048544406890869140625e-1), -0.2026592921724753704129022027337835687888e-1L, eps * 1000, __LINE__);

   check_close_l(fmaxl(0.1L, -0.1L), 0.1L, 0, __LINE__);
   check_close_l(fminl(0.1L, -0.1L), -0.1L, 0, __LINE__);

   check_close_l(hypotl(1.0L, 3.0L), sqrtl(10.0L), eps * 500, __LINE__);

   check_close_l(lgammal(3.5), 1.2009736023470742248160218814507129957702389154682L, 5000 * eps, __LINE__);
   check_close_l(lgammal(0.125), 2.0194183575537963453202905211670995899482809521344L, 5000 * eps, __LINE__);
   check_close_l(lgammal(-0.125), 2.1653002489051702517540619481440174064962195287626L, 5000 * eps, __LINE__);
   check_close_l(lgammal(-3.125), 0.1543111276840418242676072830970532952413339012367L, 5000 * eps, __LINE__);
   check_close_l(lgammal(-53249.0/1024), -149.43323093420259741100038126078721302600128285894L, 5000 * eps, __LINE__);

   check_close_l(tgammal(3.5), 3.3233509704478425511840640312646472177454052302295L, 5000 * eps, __LINE__);
   check_close_l(tgammal(0.125), 7.5339415987976119046992298412151336246104195881491L, 5000 * eps, __LINE__);
   check_close_l(tgammal(-0.125), -8.7172188593831756100190140408231437691829605421405L, 5000 * eps, __LINE__);
   check_close_l(tgammal(-3.125), 1.1668538708507675587790157356605097019141636072094L, 5000 * eps, __LINE__);

#ifdef BOOST_HAS_LONG_LONG
   check(llroundl(2.5L) == 3LL, __LINE__);
   check(llroundl(2.25L) == 2LL, __LINE__);
#endif
   check(lroundl(2.5L) == 3.0L, __LINE__);
   check(lroundl(2.25L) == 2.0L, __LINE__);
   check(roundl(2.5L) == 3.0L, __LINE__);
   check(roundl(2.25L) == 2.0L, __LINE__);

   check(nextafterl(1.0L, 2.0L) > 1.0L, __LINE__);
   check(nextafterl(1.0L, -2.0L) < 1.0L, __LINE__);
   check(nextafterl(nextafterl(1.0L, 2.0L), -2.0L) == 1.0L, __LINE__);
   check(nextafterl(nextafterl(1.0L, -2.0L), 2.0L) == 1.0L, __LINE__);
   check(nextafterl(1.0L, 2.0L) > 1.0L, __LINE__);
   check(nextafterl(1.0L, -2.0L) < 1.0L, __LINE__);
   check(nextafterl(nextafterl(1.0L, 2.0L), -2.0L) == 1.0L, __LINE__);
   check(nextafterl(nextafterl(1.0L, -2.0L), 2.0L) == 1.0L, __LINE__);

   check(truncl(2.5L) == 2.0L, __LINE__);
   check(truncl(2.25L) == 2.0L, __LINE__);

   //
   // Now for the TR1 math functions:
   //
   check_close_fraction_l(assoc_laguerrel(4, 5, 0.5L), 88.31510416666666666666666666666666666667L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_laguerrel(10, 0, 2.5L), -0.8802526766660982969576719576719576719577L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_laguerrel(10, 1, 4.5L), 1.564311458042689732142857142857142857143L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_laguerrel(10, 6, 8.5L), 20.51596541066649098875661375661375661376L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_laguerrel(10, 12, 12.5L), -199.5560968456234671241181657848324514991L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_laguerrel(50, 40, 12.5L), -4.996769495006119488583146995907246595400e16L, eps * 100, __LINE__);

   check_close_fraction_l(laguerrel(1, 0.5L), 0.5L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(4, 0.5L), -0.3307291666666666666666666666666666666667L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(7, 0.5L), -0.5183392237103174603174603174603174603175L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(20, 0.5L), 0.3120174870800154148915399248893113634676L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(50, 0.5L), -0.3181388060269979064951118308575628226834L, eps * 100, __LINE__);

   check_close_fraction_l(laguerrel(1, -0.5L), 1.5L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(4, -0.5L), 3.835937500000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(7, -0.5L), 7.950934709821428571428571428571428571429L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(20, -0.5L), 76.12915699869631476833699787070874048223L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(50, -0.5L), 2307.428631277506570629232863491518399720L, eps * 100, __LINE__);

   check_close_fraction_l(laguerrel(1, 4.5L), -3.500000000000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(4, 4.5L), 0.08593750000000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(7, 4.5L), -1.036928013392857142857142857142857142857L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(20, 4.5L), 1.437239150257817378525582974722170737587L, eps * 100, __LINE__);
   check_close_fraction_l(laguerrel(50, 4.5L), -0.7795068145562651416494321484050019245248L, eps * 100, __LINE__);

   check_close_fraction_l(assoc_legendrel(4, 2, 0.5L), 4.218750000000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_legendrel(7, 5, 0.5L), 5696.789530152175143607977274672800795328L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_legendrel(4, 2, -0.5L), 4.218750000000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(assoc_legendrel(7, 5, -0.5L), 5696.789530152175143607977274672800795328L, eps * 100, __LINE__);

   check_close_fraction_l(legendrel(1, 0.5L), 0.5L, eps * 100, __LINE__);
   check_close_fraction_l(legendrel(4, 0.5L), -0.2890625000000000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(legendrel(7, 0.5L), 0.2231445312500000000000000000000000000000L, eps * 100, __LINE__);
   check_close_fraction_l(legendrel(40, 0.5L), -0.09542943523261546936538467572384923220258L, eps * 100, __LINE__);

   sv = eps / 1024;
   check_close_l(betal(1, 1), 1, eps * 20 * 100, __LINE__);
   check_close_l(betal(1, 4), 0.25, eps * 20 * 100, __LINE__);
   check_close_l(betal(4, 1), 0.25, eps * 20 * 100, __LINE__);
   check_close_l(betal(sv, 4), 1/sv, eps * 20 * 100, __LINE__);
   check_close_l(betal(4, sv), 1/sv, eps * 20 * 100, __LINE__);
   check_close_l(betal(4, 20), 0.00002823263692828910220214568040654997176736L, eps * 20 * 100, __LINE__);
   check_close_l(betal(0.0125L, 0.000023L), 43558.24045647538375006349016083320744662L, eps * 20 * 100, __LINE__);

   check_close_l(comp_ellint_1l(0), 1.5707963267948966192313216916397514420985846996876L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(0.125), 1.5769867712158131421244030532288080803822271060839L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(0.25), 1.5962422221317835101489690714979498795055744578951L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(300/1024.0L), 1.6062331054696636704261124078746600894998873503208L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(400/1024.0L), 1.6364782007562008756208066125715722889067992997614L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(-0.5), 1.6857503548125960428712036577990769895008008941411L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_1l(-0.75), 1.9109897807518291965531482187613425592531451316788L, eps * 5000, __LINE__);

   check_close_l(comp_ellint_2l(-1), 1.0L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(0), 1.5707963267948966192313216916397514420985846996876L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(100 / 1024.0L), 1.5670445330545086723323795143598956428788609133377L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(200 / 1024.0L), 1.5557071588766556854463404816624361127847775545087L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(300 / 1024.0L), 1.5365278991162754883035625322482669608948678755743L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(400 / 1024.0L), 1.5090417763083482272165682786143770446401437564021L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(-0.5), 1.4674622093394271554597952669909161360253617523272L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(-600 / 1024.0L), 1.4257538571071297192428217218834579920545946473778L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(-800 / 1024.0L), 1.2927868476159125056958680222998765985004489572909L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_2l(-900 / 1024.0L), 1.1966864890248739524112920627353824133420353430982L, eps * 5000, __LINE__);

   check_close_l(comp_ellint_3l(0.2L, 0), 1.586867847454166237308008033828114192951L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(0.4L, 0), 1.639999865864511206865258329748601457626L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(0.0L, 0), 1.57079632679489661923132169163975144209858469968755291048747L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(0.0L, 0.5), 2.221441469079183123507940495030346849307L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(0.3L, -4), 0.712708870925620061597924858162260293305195624270730660081949L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(-0.5L, -1e+05), 0.00496944596485066055800109163256108604615568144080386919012831L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(-0.75L, -1e+10), 0.0000157080225184890546939710019277357161497407143903832703317801L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(-0.875L, 1 / 1024.0L), 2.18674503176462374414944618968850352696579451638002110619287L, eps * 5000, __LINE__);
   check_close_l(comp_ellint_3l(-0.875L, 1023/1024.0L), 101.045289804941384100960063898569538919135722087486350366997L, eps * 5000, __LINE__);

   check_close_l(cyl_bessel_il(2.25L, 1/(1024.0L*1024.0L)), 2.34379212133481347189068464680335815256364262507955635911656e-15L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(5.5L, 3.125), 0.0583514045989371500460946536220735787163510569634133670181210L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(-5 + 1.0L/1024.0L, 2.125), 0.0267920938009571023702933210070984416052633027166975342895062L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(-5.5L, 10), 597.577606961369169607937419869926705730305175364662688426534L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(-10486074.0L/(1024.0L*1024), 1/1024.0L), 1.41474005665181350367684623930576333542989766867888186478185e35L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(-10486074.0L/(1024.0L*1024), 50), 1.07153277202900671531087024688681954238311679648319534644743e20L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(144794.0L/1024.0L, 100), 2066.27694757392660413922181531984160871678224178890247540320L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_il(-144794.0L/1024.0L, 100), 2066.27694672763190927440969155740243346136463461655104698748L, eps * 5000, __LINE__);

   check_close_l(cyl_bessel_jl(2457/1024.0L, 1/1024.0L), 3.80739920118603335646474073457326714709615200130620574875292e-9L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(5.5L, 3217.0L/1024), 0.0281933076257506091621579544064767140470089107926550720453038L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-5.5L, 3217.0L/1024), -2.55820064470647911823175836997490971806135336759164272675969L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-5.5L, 1e+04L), 2.449843111985605522111159013846599118397e-03L, eps * 50000, __LINE__);
   check_close_l(cyl_bessel_jl(5.5L, 1e+04L), 0.00759343502722670361395585198154817047185480147294665270646578L, eps * 5000, __LINE__);
   //check_close_l(cyl_bessel_jl(5.5L, 1e+06), -0.000747424248595630177396350688505919533097973148718960064663632L, eps * 5000, __LINE__);
   //check_close_l(cyl_bessel_jl(5.125L, 1e+06), -0.000776600124835704280633640911329691642748783663198207360238214L, eps * 5000, __LINE__);
   //check_close_l(cyl_bessel_jl(5.875L, 1e+06), -0.000466322721115193071631008581529503095819705088484386434589780L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(0.5L, 101), 0.0358874487875643822020496677692429287863419555699447066226409L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-5.5L, 1e+04L), 0.00244984311198560552211115901384659911839737686676766460822577L, eps * 50000, __LINE__);
   //check_close_l(cyl_bessel_jl(-5.5L, 1e+06), 0.000279243200433579511095229508894156656558211060453622750659554L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-0.5L, 101), 0.0708184798097594268482290389188138201440114881159344944791454L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-10486074 / (1024*1024.0L), 1/1024.0L), 1.41474013160494695750009004222225969090304185981836460288562e35L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-10486074 / (1024*1024.0L), 15), -0.0902239288885423309568944543848111461724911781719692852541489L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(10486074 / (1024*1024.0L), 1e+02L), -0.0547064914615137807616774867984047583596945624129838091326863L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(10486074 / (1024*1024.0L), 2e+04L), -0.00556783614400875611650958980796060611309029233226596737701688L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_jl(-10486074 / (1024*1024.0L), 1e+02L), -0.0547613660316806551338637153942604550779513947674222863858713L, eps * 5000, __LINE__);

   check_close_l(cyl_bessel_kl(0.5L, 0.875), 0.558532231646608646115729767013630967055657943463362504577189L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(0.5L, 1.125), 0.383621010650189547146769320487006220295290256657827220786527L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(2.25L, ldexpl(1.0L, -30)), 5.62397392719283271332307799146649700147907612095185712015604e20L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(5.5L, 3217/1024.0L), 1.30623288775012596319554857587765179889689223531159532808379L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(-5.5L, 10), 0.0000733045300798502164644836879577484533096239574909573072142667L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(-5.5L, 100), 5.41274555306792267322084448693957747924412508020839543293369e-45L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(10240/1024.0L, 1/1024.0L), 2.35522579263922076203415803966825431039900000000993410734978e38L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(10240/1024.0L, 10), 0.00161425530039067002345725193091329085443750382929208307802221L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(144793/1024.0L, 100), 1.39565245860302528069481472855619216759142225046370312329416e-6L, eps * 5000, __LINE__);
   check_close_l(cyl_bessel_kl(144793/1024.0L, 200), 9.11950412043225432171915100042647230802198254567007382956336e-68L, eps * 5000, __LINE__);

   check_close_l(cyl_neumannl(0.5L, 1 / (1024.0L*1024)), -817.033790261762580469303126467917092806755460418223776544122L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(5.5L, 3.125), -2.61489440328417468776474188539366752698192046890955453259866L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(-5.5L, 3.125), -0.0274994493896489729948109971802244976377957234563871795364056L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(-5.5L, 1e+04), -0.00759343502722670361395585198154817047185480147294665270646578L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(-10486074 / (1024*1024.0L), 1/1024.0L), -1.50382374389531766117868938966858995093408410498915220070230e38L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(-10486074 / (1024*1024.0L), 1e+02L), 0.0583041891319026009955779707640455341990844522293730214223545L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(141.75L, 1e+02), -5.38829231428696507293191118661269920130838607482708483122068e9L, eps * 5000, __LINE__);
   check_close_l(cyl_neumannl(141.75L, 2e+04), -0.00376577888677186194728129112270988602876597726657372330194186L, eps * 50000, __LINE__);
   check_close_l(cyl_neumannl(-141.75L, 1e+02), -3.81009803444766877495905954105669819951653361036342457919021e9L, eps * 5000, __LINE__);

   check_close_l(ellint_1l(0, 0), 0, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0, -10), -10, eps * 5000, __LINE__);
   check_close_l(ellint_1l(-1, -1), -1.2261911708835170708130609674719067527242483502207L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.875L, -4), -5.3190556182262405182189463092940736859067548232647L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(-0.625L, 8), 9.0419973860310100524448893214394562615252527557062L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.875L, 1e-05L), 0.000010000000000127604166668510945638036143355898993088L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(10/1024.0L, 1e+05L), 100002.38431454899771096037307519328741455615271038L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(1, 1e-20L), 1.0000000000000000000000000000000000000000166666667e-20L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(1e-20L, 1e-20L), 1.000000000000000e-20L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(400/1024.0L, 1e+20L), 1.0418143796499216839719289963154558027005142709763e20L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, 2), 2.1765877052210673672479877957388515321497888026770L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, 4), 4.2543274975235836861894752787874633017836785640477L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, 6), 6.4588766202317746302999080620490579800463614807916L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, 10), 10.697409951222544858346795279378531495869386960090L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, -2), -2.1765877052210673672479877957388515321497888026770L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, -4), -4.2543274975235836861894752787874633017836785640477L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, -6), -6.4588766202317746302999080620490579800463614807916L, eps * 5000, __LINE__);
   check_close_l(ellint_1l(0.5L, -10), -10.697409951222544858346795279378531495869386960090L, eps * 5000, __LINE__);

   check_close_l(ellint_2l(0, 0), 0, eps * 5000, __LINE__);
   check_close_l(ellint_2l(0, -10), -10, eps * 5000, __LINE__);
   check_close_l(ellint_2l(-1, -1), -0.84147098480789650665250232163029899962256306079837L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(900 / 1024.0L, -4), -3.1756145986492562317862928524528520686391383168377L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(-600 / 1024.0L, 8), 7.2473147180505693037677015377802777959345489333465L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(800 / 1024.0L, 1e-05L), 9.999999999898274739584436515967055859383969942432E-6L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(100 / 1024.0L, 1e+05L), 99761.153306972066658135668386691227343323331995888L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(-0.5L, 1e+10L), 9.3421545766487137036576748555295222252286528414669e9L, eps * 5000, __LINE__);
   check_close_l(ellint_2l(400 / 1024.0L, ldexpl(1, 66)), 7.0886102721911705466476846969992069994308167515242e19L, eps * 5000, __LINE__);

   check_close_l(ellint_3l(0, 1, -1), -1.557407724654902230506974807458360173087L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.4L, 0, -4), -4.153623371196831087495427530365430979011L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(-0.6L, 0, 8), 8.935930619078575123490612395578518914416L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.25L, 0, 0.5L), 0.501246705365439492445236118603525029757890291780157969500480L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 0, 0.5L), 0.5L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, -2, 0.5L), 0.437501067017546278595664813509803743009132067629603474488486L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 0.25L, 0.5L), 0.510269830229213412212501938035914557628394166585442994564135L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 0.75L, 0.5L), 0.533293253875952645421201146925578536430596894471541312806165L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 0.75L, 0.75), 0.871827580412760575085768367421866079353646112288567703061975L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 1, 0.25L), 0.255341921221036266504482236490473678204201638800822621740476L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 2, 0.25L), 0.261119051639220165094943572468224137699644963125853641716219L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0, 1023/1024.0L, 1.5), 13.2821612239764190363647953338544569682942329604483733197131L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.5L, 0.5L, -1), -1.228014414316220642611298946293865487807L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.5L, 0.5L, 1e+10L), 1.536591003599172091573590441336982730551e+10L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.75L, -1e+05L, 10), 0.0347926099493147087821620459290460547131012904008557007934290L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.875L, -1e+10L, 10), 0.000109956202759561502329123384755016959364346382187364656768212L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.875L, -1e+10L, 1e+20L), 1.00000626665567332602765201107198822183913978895904937646809e15L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.875L, -1e+10L, 1608/1024.0L), 0.0000157080616044072676127333183571107873332593142625043567690379L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.875L, 1-1 / 1024.0L, 1e+20L), 6.43274293944380717581167058274600202023334985100499739678963e21L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.25L, 50, 0.1L), 0.124573770342749525407523258569507331686458866564082916835900L, eps * 5000, __LINE__);
   check_close_l(ellint_3l(0.25L, 1.125L, 1), 1.77299767784815770192352979665283069318388205110727241629752L, eps * 5000, __LINE__);

   check_close_l(expintl(1/1024.0L), -6.35327933972759151358547423727042905862963067106751711596065L, eps * 5000, __LINE__);
   check_close_l(expintl(0.125), -1.37320852494298333781545045921206470808223543321810480716122L, eps * 5000, __LINE__);
   check_close_l(expintl(0.5), 0.454219904863173579920523812662802365281405554352642045162818L, eps * 5000, __LINE__);
   check_close_l(expintl(1), 1.89511781635593675546652093433163426901706058173270759164623L, eps * 5000, __LINE__);
   check_close_l(expintl(50.5), 1.72763195602911805201155668940185673806099654090456049881069e20L, eps * 5000, __LINE__);
   check_close_l(expintl(-1/1024.0L), -6.35523246483107180261445551935803221293763008553775821607264L, eps * 5000, __LINE__);
   check_close_l(expintl(-0.125), -1.62342564058416879145630692462440887363310605737209536579267L, eps * 5000, __LINE__);
   check_close_l(expintl(-0.5), -0.559773594776160811746795939315085235226846890316353515248293L, eps * 5000, __LINE__);
   check_close_l(expintl(-1), -0.219383934395520273677163775460121649031047293406908207577979L, eps * 5000, __LINE__);
   check_close_l(expintl(-50.5), -2.27237132932219350440719707268817831250090574830769670186618e-24L, eps * 5000, __LINE__);

   check_close_fraction_l(hermitel(0, 1), 1.L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(1, 1), 2.L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(1, 2), 4.L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(1, 10), 20, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(1, 100), 200, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(1, 1e6), 2e6L, 100 * eps, __LINE__);
   //check_close_fraction_l(hermitel(10, 30), 5.896624628001300E+17L, 100 * eps, __LINE__);
   //check_close_fraction_l(hermitel(10, 1000), 1.023976960161280E+33L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(10, 10), 8.093278209760000E+12L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(10, -10), 8.093278209760000E+12L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(3, -10), -7.880000000000000E+3L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(3, -1000), -7.999988000000000E+9L, 100 * eps, __LINE__);
   check_close_fraction_l(hermitel(3, -1000000), -7.999999999988000E+18L, 100 * eps, __LINE__);

   check_close_l(riemann_zetal(0.125), -0.63277562349869525529352526763564627152686379131122L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(1023 / 1024.0L), -1023.4228554489429786541032870895167448906103303056L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(1025 / 1024.0L), 1024.5772867695045940578681624248887776501597556226L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(0.5L), -1.46035450880958681288949915251529801246722933101258149054289L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(1.125L), 8.5862412945105752999607544082693023591996301183069L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(2), 1.6449340668482264364724151666460251892189499012068L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(3.5L), 1.1267338673170566464278124918549842722219969574036L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(4), 1.08232323371113819151600369654116790277475095191872690768298L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(4 + 1 / 1024.0L), 1.08225596856391369799036835439238249195298434901488518878804L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(4.5L), 1.05470751076145426402296728896028011727249383295625173068468L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(6.5L), 1.01200589988852479610078491680478352908773213619144808841031L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(7.5L), 1.00582672753652280770224164440459408011782510096320822989663L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(8.125L), 1.0037305205308161603183307711439385250181080293472L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(16.125L), 1.0000140128224754088474783648500235958510030511915L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(0), -0.5L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-0.125L), -0.39906966894504503550986928301421235400280637468895L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-1), -0.083333333333333333333333333333333333333333333333333L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-2), 0, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-2.5L), 0.0085169287778503305423585670283444869362759902200745L, eps * 5000 * 3, __LINE__);
   check_close_l(riemann_zetal(-3), 0.0083333333333333333333333333333333333333333333333333L, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-4), 0, eps * 5000, __LINE__);
   check_close_l(riemann_zetal(-20), 0, eps * 5000 * 100, __LINE__);
   check_close_l(riemann_zetal(-21), -281.46014492753623188405797101449275362318840579710L, eps * 5000 * 100, __LINE__);
   check_close_l(riemann_zetal(-30.125L), 2.2762941726834511267740045451463455513839970804578e7L, eps * 5000 * 100, __LINE__);

   check_close_l(sph_bessell(0, 0.1433600485324859619140625e-1L), 0.9999657468461303487880990241993035937654e0L,  eps * 5000 * 100, __LINE__);
   check_close_l(sph_bessell(0, 0.1760916970670223236083984375e-1L), 0.9999483203249623334100130061926184665364e0L,  eps * 5000 * 100, __LINE__);
   check_close_l(sph_bessell(2, 0.1433600485324859619140625e-1L), 0.1370120120703995134662099191103188366059e-4L,  eps * 5000 * 100, __LINE__);
   check_close_l(sph_bessell(2, 0.1760916970670223236083984375e-1L), 0.2067173265753174063228459655801741280461e-4L,  eps * 5000 * 100, __LINE__);
   check_close_l(sph_bessell(7, 0.1252804412841796875e3L), 0.7887555711993028736906736576314283291289e-2L,  eps * 5000 * 100, __LINE__);
   check_close_l(sph_bessell(7, 0.25554705810546875e3L), -0.1463292767579579943284849187188066532514e-2L,  eps * 5000 * 100, __LINE__);

   check_close_l(sph_neumannl(0, 0.408089816570281982421875e0L), -0.2249212131304610409189209411089291558038e1L, eps * 5000 * 100, __LINE__);
   check_close_l(sph_neumannl(0, 0.6540834903717041015625e0L), -0.1213309779166084571756446746977955970241e1L,   eps * 5000 * 100, __LINE__);
   check_close_l(sph_neumannl(2, 0.408089816570281982421875e0L), -0.4541702641837159203058389758895634766256e2L,   eps * 5000 * 100, __LINE__);
   check_close_l(sph_neumannl(2, 0.6540834903717041015625e0L), -0.1156112621471167110574129561700037138981e2L,   eps * 5000 * 100, __LINE__);
   check_close_l(sph_neumannl(10, 0.1097540378570556640625e1L), -0.2427889658115064857278886600528596240123e9L,   eps * 5000 * 100, __LINE__);
   check_close_l(sph_neumannl(10, 0.30944411754608154296875e1L), -0.3394649246350136450439882104151313759251e4L,   eps * 5000 * 100, __LINE__);

   check_close_fraction_l(sph_legendrel(3, 2, 0.5L), 0.2061460599687871330692286791802688341213L, eps * 5000, __LINE__);
   check_close_fraction_l(sph_legendrel(40, 15, 0.75L), -0.406036847302819452666908966769096223205057182668333862900509L, eps * 5000, __LINE__);
#endif
#endif
}

int main(int argc, char* argv[])
{
#ifndef TEST_LD
   test_values_f("float");
   test_values("double");
#else
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_valuesl("long double");
#endif
#endif
   return errors;
}
