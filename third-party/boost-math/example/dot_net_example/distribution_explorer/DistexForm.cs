using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace distribution_explorer
{
  /// <summary>
  /// Main distribution explorer.
  /// </summary>
  public partial class DistexForm : Form
  {

    EventLog log = new EventLog();
    /// <summary>
    /// Main form
    /// </summary>
    public DistexForm()
    {
      if (!EventLog.SourceExists("EventLogDistex"))
      {
        EventLog.CreateEventSource("EventLogDistex", "Application");
      }
      log.Source = "EventLogDistex";
      log.WriteEntry("DistexForm");

      InitializeComponent();

      Application.DoEvents();
    }

    private void Form_Load(object sender, EventArgs e)
    { // Load distribution & parameters names, and default values.
      try
      {
        // Create and show splash screen:
        this.Hide();
        distexSplash frmSplash = new distexSplash();
        frmSplash.Show();
        frmSplash.Update();
        // Now load our data while the splash is showing:
        if (boost_math.any_distribution.size() <= 0)
        {
          MessageBox.Show("Problem loading any distributions, size = " + boost_math.any_distribution.size().ToString());
        }
        for (int i = 0; i < boost_math.any_distribution.size(); ++i)
        {
          distribution.Items.Add(boost_math.any_distribution.distribution_name(i));
        }
        distribution.SelectedIndex = 0; // 1st in array, but could be any other.
        // All parameters are made zero by default, but updated from chosen distribution.
        parameter1.Text = boost_math.any_distribution.first_param_default(0).ToString();
        parameter2.Text = boost_math.any_distribution.second_param_default(0).ToString();
        parameter3.Text = boost_math.any_distribution.third_param_default(0).ToString();
        //
        // Sleep and then close splash;
        Thread.Sleep(3000);
        frmSplash.Close();
        this.Visible = true;
      }
      catch
      { //
        log.WriteEntry("DistexForm_load exception!");
        MessageBox.Show("Problem loading distributions, size = " + boost_math.any_distribution.size().ToString());
      }
    }

    private void distribution_SelectedIndexChanged(object sender, EventArgs e)
    {
      int i = distribution.SelectedIndex; // distribution tab.
      parameter1Label.Text = boost_math.any_distribution.first_param_name(i);
      parameterLabel1.Text = boost_math.any_distribution.first_param_name(i); // properties tab.
      parameter2Label.Text = boost_math.any_distribution.second_param_name(i);
      parameter3Label.Text = boost_math.any_distribution.third_param_name(i);
      if (boost_math.any_distribution.first_param_name(i).Length.CompareTo(0) != 0)
      { // Actually all the distributions have at least one parameters,
        parameter1.Visible = true; // so should always be true.
        parameterLabel1.Visible = true;
      }
      else
      { // If distribution chosen has no parameter name(s) then hide.
        parameter1.Visible = false;
        parameterLabel1.Visible = false;
      }
      parameter1.Text = boost_math.any_distribution.first_param_default(i).ToString();
      // Update parameter default to match distribution.
      if (boost_math.any_distribution.second_param_name(i).Length.CompareTo(0) != 0)
      {
        parameter2.Visible = true;
        parameterLabel2.Visible = true;
        parameter2ValueLabel.Visible = true;
      }
      else
      { // hide
        parameter2.Visible = false;
        parameterLabel2.Visible = false;
        parameter2ValueLabel.Visible = false;

      }
      parameter2.Text = boost_math.any_distribution.second_param_default(i).ToString();
      if (boost_math.any_distribution.third_param_name(i).Length.CompareTo(0) != 0)
      {
        parameter3.Visible = true;
        parameterLabel3.Visible = true;
        parameter3ValueLabel.Visible = true;
      }
      else
      { // hide
        parameter3.Visible = false;
        parameterLabel3.Visible = false;
        parameter3ValueLabel.Visible = false;
      }
      parameter3.Text = boost_math.any_distribution.third_param_default(i).ToString();
      // Update tool tips to show total and supported ranges.
      PropertiesTabPage.ToolTipText = "Shows properties and ranges of chosen distribution.";
    }

    private boost_math.any_distribution dist;

    private void dataGridView1_CellEndEdit(object sender, DataGridViewCellEventArgs e)
    { // Display a grid of pdf, cdf... values from user's random variate x value.
      try
      {
        if (e.ColumnIndex == 0)
        { // Clicked on left-most random variate x column to enter a value.
          int i = e.RowIndex;
          string s = CDF_data.Rows[i].Cells[0].Value.ToString();
          double x = double.Parse(s); // Get value of users random variate x.
          double pdf = dist.pdf(x); // Compute pdf values from x
          double cdf = dist.cdf(x); // & cdf
          double ccdf = dist.ccdf(x); // & complements.
          CDF_data.Rows[i].Cells[1].Value = pdf; // and display values.
          CDF_data.Rows[i].Cells[2].Value = cdf;
          CDF_data.Rows[i].Cells[3].Value = ccdf;
        }
      }
      catch (SystemException se)
      {
          MessageBox.Show("Error in random variable value: " + se.Message, "Calculation Error");
      }
    }

    private void tabPage2_Enter(object sender, EventArgs e)
    { // Properties tab shows distribution's mean, mode, median...
      try
      { // Show chosen distribution name, and parameter names & values.
        int i = distribution.SelectedIndex;
        distributionValueLabel.Text = boost_math.any_distribution.distribution_name(i).ToString();
        parameterLabel1.Text = boost_math.any_distribution.first_param_name(i).ToString();
        parameter1ValueLabel.Text = double.Parse(parameter1.Text).ToString();
        parameterLabel2.Text = boost_math.any_distribution.second_param_name(i).ToString();
        parameter2ValueLabel.Text = double.Parse(parameter2.Text).ToString();
        parameterLabel3.Text = boost_math.any_distribution.third_param_name(i).ToString();
        parameter3ValueLabel.Text = double.Parse(parameter3.Text).ToString();

        // Show computed properties of distribution.
        try
        {
            mean.Text = dist.mean().ToString();
        }
        catch
        {
            mean.Text = "Undefined.";
        }
        try
        {
            mode.Text = dist.mode().ToString();
        }
        catch
        {
            mode.Text = "Undefined.";
        }
        try
        {
            median.Text = dist.median().ToString();
        }
        catch
        {
            median.Text = "Undefined.";
        }
        try
        {
            variance.Text = dist.variance().ToString();
        }
        catch
        {
            variance.Text = "Undefined.";
        }
        try
        {
            standard_deviation.Text = dist.standard_deviation().ToString();
        }
        catch
        {
            standard_deviation.Text = "Undefined.";
        }
        try
        {
            skewness.Text = dist.skewness().ToString();
        }
        catch
        {
            skewness.Text = "Undefined.";
        }
        try
        {
            kurtosis.Text = dist.kurtosis().ToString();
        }
        catch
        {
            kurtosis.Text = "Undefined.";
        }
        try
        {
            kurtosis_excess.Text = dist.kurtosis_excess().ToString();
        }
        catch
        {
            kurtosis_excess.Text = "Undefined.";
        }
        try
        {
            coefficient_of_variation.Text = dist.coefficient_of_variation().ToString();
        }
        catch
        {
            coefficient_of_variation.Text = "Undefined.";
        }

        rangeLowestLabel.Text = dist.lowest().ToString();
        rangeGreatestLabel.Text = dist.uppermost().ToString();
        supportLowerLabel.Text = dist.lower().ToString();
        supportUpperLabel.Text = dist.upper().ToString();
        cdfTabPage.ToolTipText = "Random variate can range from " + rangeLowestLabel.Text
          + " to " + rangeGreatestLabel.Text
          + ",\nbut is said to be supported from " + supportLowerLabel.Text
          + " to " + supportUpperLabel.Text
          + "\nWithin this supported range the PDF and CDF have values between 0 and 1,\nbut below " + supportLowerLabel.Text + " both are zero, and above "
          +  supportUpperLabel.Text + " both are unity";
      }
      catch (SystemException se)
      {
        MessageBox.Show(se.Message, "Calculation Error!");
      }
    }

    private void properties_tab_Deselecting(object sender, TabControlCancelEventArgs e)
    {
      try
      {
        if (e.TabPageIndex == 0)
        {   // Update selected distribution object:
          double x = double.Parse(parameter1.Text);
          double y = double.Parse(parameter2.Text);
          double z = double.Parse(parameter3.Text);
          int i = distribution.SelectedIndex;
          dist = new boost_math.any_distribution(i, x, y, z);
          // Clear existing CDF data (has to be a better way?):
          while (CDF_data.Rows.Count > 1)
          {
            CDF_data.Rows.Remove(CDF_data.Rows[0]);
          }
          // Clear existing quantile data (has to be a better way?):
          while (QuantileData.Rows.Count > 1)
          {
            QuantileData.Rows.Remove(QuantileData.Rows[0]);
          }
        }
      }
      catch (SystemException se)
      {
          MessageBox.Show(se.Message +
              " Please check the distribution's parameters and try again.", "Distribution Error");
          this.propertiesTab.SelectedIndex = 0;
          e.Cancel = true;
      }
    }

    private void QuantileData_CellEndEdit(object sender, DataGridViewCellEventArgs e)
    { // aka Risk & critical values tab.
      try
      {
        if (e.ColumnIndex == 0)
        {
          int i = e.RowIndex;
          string s = QuantileData.Rows[i].Cells[0].Value.ToString();
          double x = double.Parse(s);
          // Remember x is alpha: 1 - the probability:
          double lcv = dist.quantile(x);
          double ucv = dist.quantile_c(x);
          QuantileData.Rows[i].Cells[1].Value = lcv;
          QuantileData.Rows[i].Cells[2].Value = ucv;
        }
      }
      catch (SystemException se)
      {
        // TODO add some proper handling here!
        MessageBox.Show("Error in probability value: " + se.Message, "Calculation Error");
      }
    }

    private void QuantileTab_Enter(object sender, EventArgs e)
    { // Evaluate critical values (quantiles) for pre-chosen risk level.
      // and then, optionally, for other user-provided risk levels.
      try
      {
        if (QuantileData.Rows.Count == 1)
        {
          // Add some defaults:
          QuantileData.Rows.Add(5); // 5 Risk levels.
          QuantileData.Rows[0].Cells[0].Value = "0.001"; // Risk values as text,
          QuantileData.Rows[0].Cells[1].Value = dist.quantile(0.001); // & as double.
          QuantileData.Rows[0].Cells[2].Value = dist.quantile_c(0.001);
          QuantileData.Rows[1].Cells[0].Value = "0.01";
          QuantileData.Rows[1].Cells[1].Value = dist.quantile(0.01); // 99% confidence.
          QuantileData.Rows[1].Cells[2].Value = dist.quantile_c(0.01);
          QuantileData.Rows[2].Cells[0].Value = "0.05";
          QuantileData.Rows[2].Cells[1].Value = dist.quantile(0.05);
          QuantileData.Rows[2].Cells[2].Value = dist.quantile_c(0.05);
          QuantileData.Rows[3].Cells[0].Value = "0.1";
          QuantileData.Rows[3].Cells[1].Value = dist.quantile(0.1);
          QuantileData.Rows[3].Cells[2].Value = dist.quantile_c(0.1);
          QuantileData.Rows[4].Cells[0].Value = "0.33333333333333333";
          QuantileData.Rows[4].Cells[1].Value = dist.quantile(0.33333333333333333);
          QuantileData.Rows[4].Cells[2].Value = dist.quantile_c(0.33333333333333333);
        }
      }
      catch (SystemException se)
      {
        // TODO add some proper handling here!
        MessageBox.Show(se.Message, "Calculation Error");
      }
    }


    private void properties_tab_SelectedIndexChanged(object sender, EventArgs e)
    {
    }

    private void tabPage1_Click(object sender, EventArgs e)
    {
    }

    private void CDF_data_CellContentClick(object sender, DataGridViewCellEventArgs e)
    {
    }

    distexAboutBox DistexAboutBox = new distexAboutBox();

    private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
    {
      DistexAboutBox.ShowDialog();
    }

    private void DistexForm_Activated(object sender, EventArgs e)
    {
    }

    /// get AssemblyDescription
    public string AssemblyDescription
    {
      get
      {
        // Get all Description attributes on this assembly
        object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyDescriptionAttribute), false);
        // If there aren't any Description attributes, return an empty string
        if (attributes.Length == 0)
          return "";
        // If there is a Description attribute, return its value
        return ((AssemblyDescriptionAttribute)attributes[0]).Description;
      }
    }

    private void saveFileDialog1_FileOk(object sender, CancelEventArgs e)
    {
      using (StreamWriter sw = new StreamWriter(this.saveFileDialog.FileName))
      { // Write distribution info and properties to file.
        sw.WriteLine( AssemblyDescription);
        sw.WriteLine("Version " + Assembly.GetExecutingAssembly().GetName().Version.ToString());
        // Get parameter names (null "" if no parameter).
        int i = distribution.SelectedIndex;
        distributionValueLabel.Text = boost_math.any_distribution.distribution_name(i).ToString();
        sw.WriteLine(distributionValueLabel.Text + " distribution");
        parameterLabel1.Text = boost_math.any_distribution.first_param_name(i).ToString();
        parameterLabel2.Text = boost_math.any_distribution.second_param_name(i).ToString();
        parameterLabel3.Text = boost_math.any_distribution.third_param_name(i).ToString();
        string separator = "\t "; // , or tab or space?
        // Write parameter name & value.
        sw.WriteLine(parameterLabel1.Text + separator + this.parameter1.Text);
        if (boost_math.any_distribution.second_param_name(i).Length.CompareTo(0) != 0)
        { // Is a 2nd parameter.
          sw.WriteLine(parameterLabel2.Text + separator +  this.parameter2.Text);
        }
        if (boost_math.any_distribution.third_param_name(i).Length.CompareTo(0) != 0)
        { // Is a 3rd parameter.
          sw.WriteLine(parameterLabel3.Text + separator + this.parameter3.Text);
        }
        sw.WriteLine();
        sw.WriteLine("Properties");
        // Show computed properties of distribution.
        double x = double.Parse(parameter1.Text);
        double y = double.Parse(parameter2.Text);
        double z = double.Parse(parameter3.Text);
        dist = new boost_math.any_distribution(i, x, y, z);
        // Note global dist might not have been calculated yet if no of the tabs clicked.
        try
        {
            mean.Text = dist.mean().ToString();
        }
        catch
        {
            mean.Text = "Undefined";
        }
        sw.WriteLine("Mean" + separator + mean.Text);
        try
        {
            mode.Text = dist.mode().ToString();
        }
        catch
        {
            mode.Text = "Undefined";
        }
        sw.WriteLine("mode" + separator + mode.Text);
        try
        {
            median.Text = dist.median().ToString();
        }
        catch
        {
            median.Text = "Undefined";
        }
        sw.WriteLine("Median" + separator + median.Text);
        try
        {
            variance.Text = dist.variance().ToString();
        }
        catch
        {
            variance.Text = "Undefined";
        }
        sw.WriteLine("Variance" + separator + variance.Text);
        try
        {
            standard_deviation.Text = dist.standard_deviation().ToString();
        }
        catch
        {
            standard_deviation.Text = "Undefined";
        }
        sw.WriteLine("Standard Deviation" + separator + standard_deviation.Text);
        try
        {
            skewness.Text = dist.skewness().ToString();
        }
        catch
        {
            skewness.Text = "Undefined";
        }
        sw.WriteLine("Skewness" + separator + skewness.Text);
        try
        {
            coefficient_of_variation.Text = dist.coefficient_of_variation().ToString();
        }
        catch
        {
            coefficient_of_variation.Text = "Undefined";
        }
        sw.WriteLine("Coefficient of variation" + separator + coefficient_of_variation.Text);
        try
        {
            kurtosis.Text = dist.kurtosis().ToString();
        }
        catch
        {
            kurtosis.Text = "Undefined";
        }
        sw.WriteLine("Kurtosis" + separator + kurtosis.Text);
        try
        {
            kurtosis_excess.Text = dist.kurtosis_excess().ToString();
        }
        catch
        {
            kurtosis_excess.Text = "Undefined";
        }
        sw.WriteLine("Kurtosis excess" + separator + kurtosis_excess.Text);
        sw.WriteLine();

        sw.WriteLine("Range from" + separator + dist.lowest().ToString() + separator +
        "to" + separator + dist.uppermost().ToString());
        sw.WriteLine("Support from " + separator + dist.lower().ToString() +separator+
        "to " + separator + dist.upper().ToString());
        sw.WriteLine();

        //
        sw.WriteLine("Quantiles");
        if (QuantileData.Rows.Count == 1)
        { // Add some defaults:
          QuantileData.Rows.Add(5); // 5 Risk levels.
          QuantileData.Rows[0].Cells[0].Value = "0.001"; // Risk values as text,
          QuantileData.Rows[0].Cells[1].Value = dist.quantile(0.001); // & as double.
          QuantileData.Rows[0].Cells[2].Value = dist.quantile_c(0.001);
          QuantileData.Rows[1].Cells[0].Value = "0.01";
          QuantileData.Rows[1].Cells[1].Value = dist.quantile(0.01); // 99% confidence.
          QuantileData.Rows[1].Cells[2].Value = dist.quantile_c(0.01);
          QuantileData.Rows[2].Cells[0].Value = "0.05";
          QuantileData.Rows[2].Cells[1].Value = dist.quantile(0.05);
          QuantileData.Rows[2].Cells[2].Value = dist.quantile_c(0.05);
          QuantileData.Rows[3].Cells[0].Value = "0.1";
          QuantileData.Rows[3].Cells[1].Value = dist.quantile(0.1);
          QuantileData.Rows[3].Cells[2].Value = dist.quantile_c(0.1);
          QuantileData.Rows[4].Cells[0].Value = "0.33333333333333333";
          QuantileData.Rows[4].Cells[1].Value = dist.quantile(0.33333333333333333);
          QuantileData.Rows[4].Cells[2].Value = dist.quantile_c(0.33333333333333333);
        }
        // else have already been calculated by entering the quantile tab.
        for (int r = 0; r < QuantileData.Rows.Count-1; r++)
        { // Show all the rows of quantiles, including any optional user values.
          sw.WriteLine(QuantileData.Rows[r].Cells[0].Value.ToString() + separator +
            QuantileData.Rows[r].Cells[1].Value.ToString() + separator +
            QuantileData.Rows[r].Cells[2].Value.ToString());
        }
        sw.WriteLine();
        sw.WriteLine("PDF, CDF & complement(s)");
        for (int r = 0; r < CDF_data.Rows.Count-1; r++)
        { // Show all the rows of pdf, cdf, including any optional user values.
          sw.WriteLine(CDF_data.Rows[r].Cells[0].Value.ToString() + separator + // x value.
            CDF_data.Rows[r].Cells[1].Value.ToString() + separator + // pdf
            CDF_data.Rows[r].Cells[2].Value.ToString() + separator + // cdf
            CDF_data.Rows[r].Cells[3].Value.ToString());// cdf complement.
        }
        sw.WriteLine();
    }

    } // saveFileDialog1_FileOk

    private void saveToolStripMenuItem_Click(object sender, EventArgs e)
    {
      this.saveFileDialog.ShowDialog();
    }

    private void saveAsToolStripMenuItem_Click(object sender, EventArgs e)
    { // Same as Save.
      this.saveFileDialog.ShowDialog();
    }

    private void contentsToolStripMenuItem_Click(object sender, EventArgs e)
    { // In lieu of proper help.
      string helpText = "\n" + AssemblyDescription +
      "\nVersion " + Assembly.GetExecutingAssembly().GetName().Version.ToString() +
      "\nA Windows utility to show the properties of distributions " +
      "\nand permit calculation of probability density (or mass) function (PDF) " +
      "\nand cumulative distribution function (CDF) and complements from values provided." +
      "\nQuantiles are also calculated for typical risk (alpha) probabilities" +
      "\nand for probabilities provided by the user." +
      "\n" +
      "\nResults can be saved to text files using Save or SaveAs." +
      "\nAll the values on the four tabs are output to the file chosen," +
      "\nand are tab separated to assist input to other programs," +
      "\nfor example, spreadsheets or text editors." +
      "\nNote: when importing to Excel, by default only 10 decimal digits are shown by Excel:" +
      "\nit is necessary to format all cells to display the full 15 decimal digits," +
      "\nalthough not all computed values will be as accurate as this." +
      "\n\nValues shown as NaN cannot be calculated for the value given," +
      "\nmost commonly because the value is outside the range for the distribution." +
      "\n" +
      "\nFor more information, including downloads, see " +
      "\nhttp://sourceforge.net/projects/distexplorer/" +
      "\n(Note that .NET framework 4.0 and VC Redistribution X86 are requirements for this program.)" +
      "\n\nCopyright John Maddock & Paul A. Bristow 2007, 2009, 2010, 2012";

      MessageBox.Show("Statistical Distribution Explorer\n" + helpText);
    }

    private void newToolStripMenuItem_Click(object sender, EventArgs e)
    {
      MessageBox.Show("New is not yet implemented.");
    }

    private void openToolStripMenuItem_Click(object sender, EventArgs e)
    {
      MessageBox.Show("Open is not yet implemented.");
    }

    private void printToolStripMenuItem_Click(object sender, EventArgs e)
    {
      MessageBox.Show("Print is not yet implemented." +
        "\nSave all values to a text file and print that file.");
    }

    private void printPreviewToolStripMenuItem_Click(object sender, EventArgs e)
    {
      MessageBox.Show("Print Preview is not yet implemented." +
        "\nSave all values to a text file and print that file.");
    }


    private void exitToolStripMenuItem_Click(object sender, EventArgs e)
    { // exit DistexForm
        this.Close();
    }
  } // class DistexForm
} // namespace distribution_explorer